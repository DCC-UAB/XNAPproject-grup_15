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
import random

torch.backends.cudnn.deterministic = True
print(os.getcwd(),'\n\n\n')

# GROUP DATASETS
IMAGE_PATH = '/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/CACD/CACD2000-centered'

# grup 1 (0, 5)
g1_TRAIN_CSV_PATH = 'datasets/separate_ages/cacd_train_grup_1.csv'
g1_TEST_CSV_PATH = 'datasets/separate_ages/cacd_test_grup_1.csv'

# grup 2 (6, 11)
g2_TRAIN_CSV_PATH = 'datasets/separate_ages/cacd_train_grup_2.csv'
g2_TEST_CSV_PATH = 'datasets/separate_ages/cacd_test_grup_2.csv'

# grup 3 (12, 17)
g3_TRAIN_CSV_PATH = 'datasets/separate_ages/cacd_train_grup_3.csv'
g3_TEST_CSV_PATH = 'datasets/separate_ages/cacd_test_grup_3.csv'

# grup 4 (18, 23)
g4_TRAIN_CSV_PATH = 'datasets/separate_ages/cacd_train_grup_4.csv'
g4_TEST_CSV_PATH = 'datasets/separate_ages/cacd_test_grup_4.csv'

# grup 5 (24, 30)
g5_TRAIN_CSV_PATH = 'datasets/separate_ages/cacd_train_grup_5.csv'
g5_TEST_CSV_PATH = 'datasets/separate_ages/cacd_test_grup_5.csv'

# grup 6 (31, 39)
g6_TRAIN_CSV_PATH = 'datasets/separate_ages/cacd_train_grup_6.csv'
g6_TEST_CSV_PATH = 'datasets/separate_ages/cacd_test_grup_6.csv'

# Argparse helper
parser = argparse.ArgumentParser()
parser.add_argument('--grup_num', type=int, default=1)

args = parser.parse_args()
print(torch.cuda.is_available(), 98989)

# select dataset
num_grup = args.grup_num
if num_grup == 1:
    TRAIN_CSV_PATH = g1_TRAIN_CSV_PATH
    TEST_CSV_PATH = g1_TEST_CSV_PATH
elif num_grup == 2:
    TRAIN_CSV_PATH = g2_TRAIN_CSV_PATH
    TEST_CSV_PATH = g2_TEST_CSV_PATH
elif num_grup == 3:
    TRAIN_CSV_PATH = g3_TRAIN_CSV_PATH
    TEST_CSV_PATH = g3_TEST_CSV_PATH
elif num_grup == 4:
    TRAIN_CSV_PATH = g4_TRAIN_CSV_PATH
    TEST_CSV_PATH = g4_TEST_CSV_PATH
elif num_grup == 5:
    TRAIN_CSV_PATH = g5_TRAIN_CSV_PATH
    TEST_CSV_PATH = g5_TEST_CSV_PATH
elif num_grup == 6:
    TRAIN_CSV_PATH = g6_TRAIN_CSV_PATH
    TEST_CSV_PATH = g6_TEST_CSV_PATH

NUM_WORKERS = 3
DEVICE = torch.device("cuda:0")
RANDOM_SEED = 452
DA_RATIO = 0.5
transform_num = 1
PATH = "model-code/separate_ages_models/CACD/BM_SEP_CACD"
BEST_MODEL_PATH = os.path.join(PATH, f'bm_CACD_g{num_grup}.pth')

##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate = 0.005
num_epochs = 45

# Architecture
NUM_CLASSES = 26   
BATCH_SIZE = 128
GRAYSCALE = False

# Wandb initialization
wandb.init(
    # set the wandb project where this run will be logged
        entity='xisca',
        project="projecte-deep-ordenat",
        name = f'cacd-rn50-bm-grup{num_grup}',
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "MSE",
            "model": "rn50",
            "dataset": "cacd-grup{num_grup}",
            "epochs": num_epochs,
            }
    )

###################
# Dataset
###################


class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self,
                 csv_path, img_dir, transform=None, augmentation_transform=None, augment_ratio=0.5):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values
        self.transform = transform
        self.augmentation_transform = augmentation_transform
        self.augment_ratio = augment_ratio
        self.normal_img_path = "/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/CACD/CACD2000"

    # def __getitem__(self, index):
       
    #     imatge = os.path.join(self.img_dir,
    #                                 self.img_names[index])
    #     img = Image.open(imatge)
    
    #     if self.transform is not None:
    #         img = self.transform(img)

    #     label = self.y[index]
    #     levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
    #     levels = torch.tensor(levels, dtype=torch.float32)

    #     return img, label, levels

    def __getitem__(self, index):
        imatge = os.path.join(self.img_dir,
                                    self.img_names[index])
        
        try:
            img = Image.open(imatge)
        except:
            imatge = os.path.join(self.normal_img_path, self.img_names[index])
            img = Image.open(imatge)

        # Apply data augmentation based on the augment_ratio
        if self.augmentation_transform and random.random() < self.augment_ratio:
            img = self.augmentation_transform(img)
        elif self.transform:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]

custom_transform = transforms.Compose([
                                        transforms.Resize((128, 128)),
                                        transforms.RandomCrop((120, 120)),
                                        transforms.ToTensor()])

train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                            img_dir=IMAGE_PATH,
                            transform=custom_transform)


custom_transform2 = transforms.Compose([ 
                                        transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])

# transform 1
augmentation_transform1 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomResizedCrop(size=(120, 120), scale=(0.8, 1.0)),
    transforms.ToTensor()
])

# transform 2
augmentation_transform2 = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop((120, 120)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
    transforms.RandomResizedCrop(size=(120, 120), scale=(0.8, 1.0)),
    transforms.ToTensor()
])

dict_transforms = {
    1: augmentation_transform1,
    2: augmentation_transform2
}

test_dataset = CACDDataset(csv_path=TEST_CSV_PATH,
                           img_dir=IMAGE_PATH,
                           transform=custom_transform2)


train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform,
                                augmentation_transform=dict_transforms[transform_num],
                                augment_ratio=DA_RATIO)

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
    model = models.resnet18(weights=None)
    
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
load_best_model_weights(model, "model-code/experiments-model-code/CACD/best_models/bm-rn18-DA-desc8-mig512-drp0.2-fc1-0.01-centered.pth")

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