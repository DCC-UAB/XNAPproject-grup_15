# coding: utf-8

#############################################
# Ordinal Regression Code with ResNet-34
#############################################

# Imports

import os
import time
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import sys
import random

# import viola jones face recognition
import cv2

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
import wandb
import torchvision.models as models


torch.backends.cudnn.deterministic = True
"""
TRAIN_CSV_PATH = './datasets/cacd_train.csv'
VALID_CSV_PATH = './datasets/cacd_valid.csv'
TEST_CSV_PATH = './cacd_test.csv'
IMAGE_PATH = '/shared_datasets/CACD/centercropped/jpg'
"""
TRAIN_CSV_PATH = "/home/xnmaster/projecte/XNAPproject-grup_15/datasets/cacd-train1.csv"
TEST_CSV_PATH = "/home/xnmaster/projecte/XNAPproject-grup_15/datasets/cacd-test1.csv"
IMAGE_PATH = '/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/CACD/CACD2000-centered'

# Argparse helper
Image.open("/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/CACD/CACD2000/28_Rebecca_Ferguson_0005.jpg")
parser = argparse.ArgumentParser()
parser.add_argument('--cuda',
                    type=int,
                    default=-1)

parser.add_argument('--seed',
                    type=int,
                    default=-1)

parser.add_argument('--numworkers',
                    type=int,
                    default=3)

parser.add_argument('--outpath',
                    type=str,
                    required=True)

# parser.add_argument('--imp_weight',
#                     type=int,
#                     default=0)

# DA args
parser.add_argument('--DA_ratio',
                    type=float,
                    default=False)

parser.add_argument('--transform_num',
                    type=int,
                    default=0)

# Freezed blocks args
parser.add_argument('--n_blocs_descongelats',
                    type=int,
                    default=1)

parser.add_argument('--n_neurones_mig',
                    type=int,
                    default=512)

parser.add_argument('--pretrain_model_path',
                    type=str,
                    default=False)

parser.add_argument('--dropout_prob',
                    type=float,
                    default=0.5)


# Save args
args = parser.parse_args()

NUM_WORKERS = args.numworkers

if args.cuda >= 0:
    DEVICE = torch.device("cuda:%d" % args.cuda)
else:
    DEVICE = torch.device("cpu")

if args.seed == -1:
    RANDOM_SEED = None
else:
    RANDOM_SEED = args.seed

if args.DA_ratio < 2 and args.DA_ratio > 0:
    DA_RATIO = args.DA_ratio
else:
    DA_RATIO = 0.5

if args.transform_num < 3 and args.transform_num > 0:
    transform_num = args.transform_num
    do_data_augmentation = True
else:
    transform_num = 0
    do_data_augmentation = False

PATH = args.outpath

if args.n_blocs_descongelats < 1:
    raise ValueError('El nombre de blocs descongelats ha de ser com a mínim 1')
else:
    N_BLOCS_DESCONGELATS = args.n_blocs_descongelats

N_NEURONES_MIG = args.n_neurones_mig

PRETRAIN_MODEL_PATH = args.pretrain_model_path

DROPOUT_PROB = args.dropout_prob

learning_rate = 0.01

if not os.path.exists(PATH):
    os.mkdir(PATH)

LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
BEST_MODEL_PATH = os.path.join(PATH, f'bm-rn18-DA-desc{N_BLOCS_DESCONGELATS}-mig{N_NEURONES_MIG}-drp{DROPOUT_PROB}-fc1-{learning_rate}-centered.pth')


# Logging

header = []

header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Output Path: %s' % PATH)
header.append('Script: %s' % sys.argv[0])
header.append('Data Augmentation: %s' % do_data_augmentation)
header.append('Transform number: %s' % transform_num)
header.append('Freezed blocks: %s' % N_BLOCS_DESCONGELATS)
header.append('Neurones mig: %s' % N_NEURONES_MIG)
header.append('Dropout prob: %s' % DROPOUT_PROB)
header.append('Pretrain model path: %s' % PRETRAIN_MODEL_PATH)


with open(LOGFILE, 'w') as f:
    for entry in header:
        print(entry)
        f.write('%s\n' % entry)
        f.flush()


##########################
# SETTINGS
##########################

# Hyperparameters
learning_rate = 0.5
num_epochs = 45


# Architecture
NUM_CLASSES = 49
BATCH_SIZE = 1024
GRAYSCALE = False

df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
ages = df['age'].values
del df
ages = torch.tensor(ages, dtype=torch.float)



# Wandb initialization
wandb.init(
    # set the wandb project where this run will be logged
        entity='xisca',
        project="projecte-deep-ordenat",
        name = f'RESNET18 {N_BLOCS_DESCONGELATS} descongelats - DA {DA_RATIO}% transf{transform_num}-pretrain_basic -centered',
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "MSE",
            "model": f"resnet50 6 descongelats - DA {DA_RATIO}% {transform_num}-fc1",
            "dataset": "cacd",
            "epochs": num_epochs,
            "pretrain": "basic IMAGENET",
            "dropout": DROPOUT_PROB,
            "vj_crop": "yes",
            }
    )

###################
# Viola Jones Face Cropping
###################

import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def violla_jones_crop(img):
    # Convert PIL Image to NumPy array
    img_np = np.array(img)
    
    # Convert RGB to BGR
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(img_bgr, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Assuming each image has exactly one face, crop the detected face
    if len(faces) > 0:
        x, y, w, h = faces[0]
        cropped_img = img_bgr[y:y+h, x:x+w]
        
        # Convert back to PIL Image
        cropped_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        return cropped_pil
    else:
        # Return the original image if no face is detected
        return img


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

if do_data_augmentation:
    train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform,
                                augmentation_transform=dict_transforms[transform_num],
                                augment_ratio=DA_RATIO)
else:
    train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                                img_dir=IMAGE_PATH,
                                transform=custom_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=NUM_WORKERS)



test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=NUM_WORKERS)


##########################
# MODEL
##########################


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def set_parameter_requires_grad_blocs(model, feature_extracting):
    if feature_extracting:
        for param in list(model.parameters())[:-N_BLOCS_DESCONGELATS]: # modificar això depenent de lo q vols descongelar
            param.requires_grad = False

# FUNCIÓ PER INIT RESNET50 PRETRAINED
def init_resnet(grayscale, pretrain_basic = True):
    if pretrain_basic:
        model = models.resnet18(weights="IMAGENET1K_V1")
    else:
        model = models.resnet18(weights=None)
    set_parameter_requires_grad(model, True)
    if grayscale:
        print('Grayscale')
        # Modifiquem 1a capa per acceptar grayscale (1 channel)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        print('No grayscale')
        # Modifiquem 1a capa per acceptar RGB (3 channel)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modificar la última capa para el número de clases especificado
    num_ftrs = model.fc.in_features
    hidden_size = N_NEURONES_MIG
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_size),
        nn.ReLU(),
        nn.Dropout(DROPOUT_PROB),
        nn.Linear(hidden_size, 1) 
    )

    return model

def load_best_model_weights(model, best_model_path):
    # Cargar los pesos del archivo best_model.pth
    pretrained_dict = torch.load(best_model_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()
    # best model torch size

    # Filtrar los pesos preentrenados para que coincidan con los nombres de las capas en el modelo definido
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Actualizar el diccionario del modelo con los pesos preentrenados
    model_dict.update(pretrained_dict)
    
    # Cargar los pesos actualizados en el modelo
    model.load_state_dict(model_dict)

    # Aplicar la función set_parameter_requires_grad para descongelar las capas necesarias
    set_parameter_requires_grad_blocs(model,True)

###########################################
# Initialize Cost, Model, and Optimizer
###########################################

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

# Inicialitzar el model Resnet50
if PRETRAIN_MODEL_PATH:
    model = init_resnet(GRAYSCALE, pretrain_basic=False)
    load_best_model_weights(model, PRETRAIN_MODEL_PATH)
else:
    model = init_resnet(GRAYSCALE)
    set_parameter_requires_grad_blocs(model, True)

model.to(DEVICE)

# afegim a la llista les capes a actualitzar (descongelades)
params_to_update = []
len_model = len(list(model.named_parameters()))
i = 0
for name, param in (model.named_parameters()):
    if param.requires_grad == True:
        params_to_update.append(param)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Métricas
def compute_mae_mse_and_r2(model, data_loader, device):
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    mae, mse, r2, num_examples = 0., 0., 0., 0
    for i, (features, targets) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device).float()

        logits= model(features)
        logits= logits.squeeze()
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


### TRAIN LOOP ###
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

        # LOGGING
        if not batch_idx % 50:
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                 % (epoch+1, num_epochs, batch_idx,
                     len(train_dataset)//BATCH_SIZE, cost))
            print(s)
            wandb.log({'Epoch': epoch+1, 'Batch': batch_idx, 'Train Loss': cost.item()})
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
        logits= model(features)
        all_probas.append(logits)
        predicted_labels = logits > 0.5
        predicted_labels = predicted_labels.int().sum(dim=1)

        lst = [str(int(i)) for i in predicted_labels]
        all_pred.extend(predicted_labels)

torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
with open(TEST_PREDICTIONS, 'w') as f:
    all_pred = ','.join(all_pred)
    f.write(all_pred)

    
    