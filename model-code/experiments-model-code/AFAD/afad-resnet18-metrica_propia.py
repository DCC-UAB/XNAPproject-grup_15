import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import torchvision.models as models

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
import wandb

torch.backends.cudnn.deterministic = True
print(os.getcwd(),'\n\n\n')
TRAIN_CSV_PATH = '/home/xnmaster/XNAPproject-grup_15-main/dataset_split/RANGE_splitted_datasets/afad_splitRANGE_train.csv'
TEST_CSV_PATH = '/home/xnmaster/XNAPproject-grup_15-main/dataset_split/RANGE_splitted_datasets/afad_splitRANGE_test.csv'
IMAGE_PATH = '/home/xnmaster/XNAPproject-grup_15-main/AFAD-Full'


# Argparse helper
parser = argparse.ArgumentParser()
parser.add_argument('--cuda',
                    type=int,
                    default=-1)

parser.add_argument('--numworkers',
                    type=int,
                    default=3)


parser.add_argument('--seed',
                    type=int,
                    default=-1)

parser.add_argument('--outpath',
                    type=str,
                    required=True)

args = parser.parse_args()
print(torch.cuda.is_available(), 98989)

NUM_WORKERS = args.numworkers

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')


# Logging

header = []

header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Output Path: %s' % PATH)
header.append('Script: %s' % sys.argv[0])

with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()


##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate = 0.0005
num_epochs = 45

# Architecture
NUM_CLASSES = 10   # canviat per fer afad amb dataset RANGE 0-9
BATCH_SIZE = 256
GRAYSCALE = False

### WANDB INITIALIZATION

wandb.login(key='eb28261147123809156d0add6be5ba7f687a0514')

# Wandb initialization
wandb.init(
    # set the wandb project where this run will be logged
        entity='xisca',
        project="projecte-deep",
        name = 'afad-metrica_propia-noves-metriques-resnet18  Range',
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "MSE",
            "model": "resnet34-pretrained",
            "dataset": "afad",
            "epochs": num_epochs,
            }
    )

###################
# Dataset
###################


class AFADDatasetAge(Dataset):
    """Custom Dataset for loading AFAD face images"""

    def __init__(self, csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_paths = df['path']
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_paths[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]

        return img, label

    def __len__(self):
        return self.y.shape[0]


custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomCrop((120, 120)),
                                       transforms.ToTensor()])

train_dataset = AFADDatasetAge(csv_path=TRAIN_CSV_PATH,
                               img_dir=IMAGE_PATH,
                               transform=custom_transform)


custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                        transforms.CenterCrop((120, 120)),
                                        transforms.ToTensor()])

test_dataset = AFADDatasetAge(csv_path=TEST_CSV_PATH,
                              img_dir=IMAGE_PATH,
                              transform=custom_transform2)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)

assert torch.cuda.is_available(), "GPU is not enabled"

##########################
# MODEL
##########################

# FUNCIÓ PER INIT RESNET34 PRETRAINED
def init_resnet34(num_classes, grayscale):
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    if grayscale:
        # Modifiquem 1a capa per acceptar grayscale (1 channel)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        # Modifiquem 1a capa per acceptar RGB (3 channel)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modificar la última capa para el número de clases especificado
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model

###########################################
# Initialize Cost, Model, and Optimizer
###########################################

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Initialize the model
model = init_resnet34(NUM_CLASSES, GRAYSCALE)
model.to(DEVICE)

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def compute_mae_and_mse(model, data_loader, device):
     mae, mse, num_examples = 0., 0., 0
     for i, (features, targets) in enumerate(data_loader):
            
         features = features.to(device)
         targets = targets.to(device)

         logits = model(features)
         probas = F.softmax(logits, dim=1)
         _, predicted_labels = torch.max(probas, 1)
         num_examples += targets.size(0)
         mae += torch.sum(torch.abs(predicted_labels - targets))
         mse += torch.sum((predicted_labels - targets)**2)
    
     mae = mae.float()/num_examples
     mse = mse.float()/num_examples
       
     return mae, mse

# Training Loop
start_time = time.time()

best_mae, best_rmse, best_r2, best_epoch = 999, 999, -1, -1
age_group_width=1
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # FORWARD AND BACK PROP
        # FORWARD AND BACK PROP
        logits = model(features)
        probas = F.softmax(logits, dim=1)

        # Calcular el costo
        predicted_labels = torch.argmax(probas, dim=1)
        correct_predictions = (predicted_labels == targets)
        differences = torch.abs(predicted_labels - targets)
        age_differences = differences * age_group_width  # age_group_width sería la diferencia en años entre grupos de edad
        cost = F.cross_entropy(logits, targets)
        incorrect_penalty = torch.sum(age_differences[~correct_predictions])  # Penalizar solo las predicciones incorrectas
        total_cost = cost + incorrect_penalty / len(targets)

        optimizer.zero_grad()
        total_cost.backward()
        optimizer.step()

        # LOGGING
        if not batch_idx % 50:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Total Cost: %.4f | Classification Cost: %.4f | Incorrect Penalty: %.4f'
                 % (epoch+1, num_epochs, batch_idx,
                    len(train_dataset)//BATCH_SIZE, total_cost, cost, incorrect_penalty))
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):
        train_mae, train_mse = compute_mae_and_mse(model, train_loader, device=DEVICE)
        test_mae, test_mse = compute_mae_and_mse(model, test_loader, device=DEVICE)


        wandb.log({
            'epoch': epoch, 
            'train_mae': train_mae, 'train_mse': train_mse, 
            'test_mae': test_mae, 'test_mse': test_mse
        })

    if test_mae < best_mae:
        best_mae, best_rmse, best_epoch = test_mae, (test_mse)**0.5, epoch

########## SAVE PREDICTIONS ######

model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
model.eval()
all_pred = []
with torch.set_grad_enabled(False):
    for batch_idx, (features, targets) in enumerate(test_loader):

        features = features.to(DEVICE)
        logits, probas = model(features)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        lst = [str(int(i)) for i in predicted_labels]
        all_pred.extend(lst)

with open(TEST_PREDICTIONS, 'w') as f:
    all_pred = ','.join(all_pred)
    f.write(all_pred)