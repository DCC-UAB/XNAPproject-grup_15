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
TRAIN_CSV_PATH = '/home/xnmaster/projecte/XNAPproject-grup_15/datasets/afad-propi-train.csv'
TEST_CSV_PATH = '/home/xnmaster/projecte/XNAPproject-grup_15/datasets/afad-propi-test.csv'
IMAGE_PATH = '/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/AFAD/AFAD-Full'

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
BEST_MODEL_PATH = os.path.join(PATH, 'best_model.pth')

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
NUM_CLASSES = 26   
BATCH_SIZE = 256
GRAYSCALE = False

# Wandb initialization
wandb.init(
    # set the wandb project where this run will be logged
        entity='xisca',
        project="projecte-deep-ordenat",
        name = 'AFAD-RESNET-50-DESCONGELANT-8-BLOCK-MENYS-FC-DATASET-PROPI-PENULTIMA-CAPA-512-NEURONES-GUARDA-MODEL',
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "MSE",
            "model": "resnet18-pretrained - fe",
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


def set_parameter_requires_grad(model):
    for param in list(model.parameters())[:-8]:
        param.requires_grad = False
    model.fc.requires_grad = True

def init_resnet50(num_classes, grayscale):
    # Inicializar el modelo ResNet50
    model = models.resnet50(weights=None)
    
    # Modificar la primera capa para aceptar imágenes en escala de grises o RGB
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Número de características en la penúltima capa
    num_ftrs = model.fc.in_features
    
    # Ajustar la penúltima capa para evitar un cambio brusco antes de la capa de salida
    hidden_size = 512
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_size, 1) 
    )

    return model

def load_best_model_weights(model, best_model_path):
    # Cargar los pesos del archivo best_model.pth
    pretrained_dict = torch.load(best_model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()

    # Filtrar los pesos preentrenados para que coincidan con los nombres de las capas en el modelo definido
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Actualizar el diccionario del modelo con los pesos preentrenados
    model_dict.update(pretrained_dict)
    
    # Cargar los pesos actualizados en el modelo
    model.load_state_dict(model_dict)

    # Aplicar la función set_parameter_requires_grad para descongelar las capas necesarias
    set_parameter_requires_grad(model)



###########################################
# Initialize Cost, Model, and Optimizer
###########################################

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Inicializar el modelo ResNet50
model = init_resnet50(NUM_CLASSES, GRAYSCALE)

# Cargar los pesos del mejor modelo
load_best_model_weights(model, "best_model.pth")

model.to(DEVICE)
params_to_update = []
len_model = len(list(model.named_parameters()))
i = 0
for name, param in (model.named_parameters()):
    if param.requires_grad == True:
        params_to_update.append(param)

# Initialize the optimizer
optimizer = torch.optim.Adam(params_to_update, lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Métricas
def compute_mae_mse_and_r2(model, data_loader, device):
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    mae, mse, r2, num_examples = 0., 0., 0., 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device).float()

        logits = model(features).squeeze()
        num_examples += targets.size(0)
        
        # Calcula el MAE y el MSE utilizando las etiquetas y las predicciones directamente
        mae += mae_loss(logits, targets).item() * features.size(0)
        mse += mse_loss(logits, targets).item() * features.size(0)
        
        # Calcula R^2
        mean_targets = torch.mean(targets)
        TSS = torch.sum((targets - mean_targets)**2)
        RSS = torch.sum((targets - logits)**2)
        r2 += 1 - (RSS / TSS)

    mae = mae / num_examples
    mse = mse / num_examples
    r2 = r2 / num_examples

    return mae, mse, r2

# Training Loop
start_time = time.time()

best_mae, best_rmse, best_r2, best_epoch = 999, 999, -1, -1
for epoch in range(num_epochs):
    model.train()

    for batch_idx, (features, targets) in enumerate(train_loader):
        features = features.to(DEVICE)
        targets = targets.to(DEVICE).float()

        optimizer.zero_grad()

        logits = model(features).squeeze()
        cost = F.mse_loss(logits, targets)

        cost.backward()
        optimizer.step()

        if not batch_idx % 50:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                 % (epoch+1, num_epochs, batch_idx,
                    len(train_dataset)//BATCH_SIZE, cost))

            wandb.log({'Epoch': epoch+1, 'Batch': batch_idx, 'Train Loss': cost.item()})
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)

    model.eval()
    with torch.set_grad_enabled(False):
        train_mae, train_mse, train_r2 = compute_mae_mse_and_r2(model, train_loader, device=DEVICE)
        test_mae, test_mse, test_r2 = compute_mae_mse_and_r2(model, test_loader, device=DEVICE)

        wandb.log({
            'epoch': epoch, 
            'train_mae': train_mae, 'train_mse': train_mse, 'train_r2': train_r2, 
            'test_mae': test_mae, 'test_mse': test_mse, 'test_r2': test_r2
        })
        
    scheduler.step()

    if test_mae < best_mae:
        best_mae, best_rmse, best_r2, best_epoch = test_mae, (test_mse)**0.5, test_r2, epoch
        torch.save(model.state_dict(), BEST_MODEL_PATH)

TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')
########## SAVE PREDICTIONS ######
all_pred = []
all_probas = []
with torch.set_grad_enabled(False):
    for batch_idx, (features, targets) in enumerate(test_loader):

        features = features.to(DEVICE)
        logits = model(features)
        all_probas.append(logits)
        predicted_labels = logits > 0.5
        predicted_labels = predicted_labels.int().sum(dim=1)

        lst = [str(int(i)) for i in predicted_labels]
        all_pred.extend(predicted_labels)

torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
with open(TEST_PREDICTIONS, 'w') as f:
    all_pred = ','.join(map(str, all_pred))
    f.write(all_pred)
