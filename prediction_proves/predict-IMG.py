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
### args ###
parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default=1)
parser.add_argument('--real_age', type=int, default=1)
parser.add_argument('--dataset', type=str, default='AFAD_EQUI')
parser.add_argument('--loss', type=str, default='mse')
args = parser.parse_args()
print(torch.cuda.is_available(), 98989)

img_path = args.img_path
r_age = args.real_age
dataset = args.dataset
loss = args.loss

##########################
# MODEL
##########################

DEVICE = torch.device("cuda:0")


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def set_parameter_requires_grad_blocs(model, feature_extracting):
    if feature_extracting:
        for param in list(model.parameters())[:-8]: # modificar això depenent de lo q vols descongelar
            param.requires_grad = False

# FUNCIÓ PER INIT RESNET50 PRETRAINED
def init_resnet_AFAD(grayscale, pretrain_basic = True):
    if pretrain_basic:
        model = models.resnet50(weights="IMAGENET1K_V1")
    else:
        model = models.resnet50(weights=None)
    set_parameter_requires_grad(model, True)
    if grayscale:
        # Modifiquem 1a capa per acceptar grayscale (1 channel)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        # Modifiquem 1a capa per acceptar RGB (3 channel)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modificar la última capa para el número de clases especificado
    num_ftrs = model.fc.in_features
    hidden_size = 512
    
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_size, 1) 
    )

    return model

# FUNCIÓ PER INIT RESNET18 PRETRAINED
def init_resnet_CACD(grayscale, pretrain_basic = True):
    if pretrain_basic:
        model = models.resnet18(weights="IMAGENET1K_V1")
    else:
        model = models.resnet18(weights=None)
    set_parameter_requires_grad(model, True)
    if grayscale:
        # Modifiquem 1a capa per acceptar grayscale (1 channel)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        # Modifiquem 1a capa per acceptar RGB (3 channel)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modificar la última capa para el número de clases especificado
    num_ftrs = model.fc.in_features
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

torch.manual_seed(400)
torch.cuda.manual_seed(400)

if loss == 'mse':
    if dataset == 'AFAD_EQUI':
        model = init_resnet_AFAD(False, pretrain_basic=False)
        load_best_model_weights(model, "/home/xnmaster/projecte/XNAPproject-grup_15/model-code/models-dataset-equilibrat/AFAD/BM_EQUI_AFAD/bm_AFAD_EQUI-b2-pretrain_equi1.pth")
        suma_age = 15

    if dataset == 'AFAD':
        model = init_resnet_AFAD(False, pretrain_basic=False)
        load_best_model_weights(model, "/home/xnmaster/projecte/XNAPproject-grup_15/model-code/experiments-model-code/Carpeta_AFAD_modelos_buenos/Congelacio_Blocs/best_model_actual-8-blocs-desc-DA0.5-1.pth")
        suma_age = 15

    if dataset == 'CACD':
        model = init_resnet_CACD(False, pretrain_basic=False)
        load_best_model_weights(model, "/home/xnmaster/projecte/XNAPproject-grup_15/model-code/experiments-model-code/CACD/best_models/bm-rn18-DA-desc2-mig512-drp0.2-fc1.pth")
        suma_age = 14

print(loss)
model.to(DEVICE)

# afegim a la llista les capes a actualitzar (descongelades)
params_to_update = []
len_model = len(list(model.named_parameters()))
i = 0
for name, param in (model.named_parameters()):
    if param.requires_grad == True:
        params_to_update.append(param)

### make single prediction

custom_transform2 = transforms.Compose([ 
                                        transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])


im = Image.open(img_path)
im = custom_transform2(im)
im = im.unsqueeze(0)
im = im.to(DEVICE)

model.eval()
with torch.no_grad():
    outputs = model(im)
    print(f'\n------------PREDICCIÓ--------------')
    print("Predicted age: ", outputs.item()+suma_age)
    print("Real age: ", r_age)
    print('--------------------------------------\n')


end_time = time.time()


    