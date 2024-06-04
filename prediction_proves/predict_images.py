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


def set_parameter_requires_grad(model):
    for param in list(model.parameters())[:-8]:
        param.requires_grad = False
    model.fc.requires_grad = True

def init_resnet50_sense(grayscale):
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




def set_parameter_requires_grad(model):
    # Congelar todos los parámetros de la red
    for param in model.parameters():
        param.requires_grad = False
    # Descongelar los parámetros de las capas fc_age y fc_gender
    for name, param in model.named_parameters():
        if 'fc_age' in name or 'fc_gender' in name:
            param.requires_grad = True


def init_resnet50_dropout(grayscale):
    # Inicializar el modelo ResNet50
    model = models.resnet50(weights='IMAGENET1K_V1')
    
    # Modificar la primera capa para aceptar imágenes en escala de grises o RGB
    if grayscale:
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    else:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Número de características en la penúltima capa
    num_ftrs = model.fc.in_features
    
    # Ajustar la penúltima capa para evitar un cambio brusco antes de la capa de salida
    hidden_size = 512
    
    model.fc_age = nn.Sequential(
        nn.Linear(num_ftrs, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_size, 1)  # Salida para la predicción de la edad (regresión)
    )

    model.fc_gender = nn.Sequential(
        nn.Linear(num_ftrs, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(hidden_size, 2)  # Salida para la predicción del género (clasificación) 
    )

    return model

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

    # Filtrar los pesos preentrenados para que coincidan con los nombres de las capas en el modelo definido
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # Actualizar el diccionario del modelo con los pesos preentrenados
    model_dict.update(pretrained_dict)
    
    # Cargar los pesos actualizados en el modelo
    model.load_state_dict(model_dict)

# Definir la función para preprocesar la imagen
def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    image = image.unsqueeze(0)  # Agregar una dimensión para el batch
    return image

# Definir la función para predecir la edad
def predict_age(model, image_tensor):
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        print(output.shape)
        age = output.item()
    return age

# take model path and image path as args
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, help="Path to the trained model")
parser.add_argument("--image_path", type=str, help="Path to the image to predict")
parser.add_argument("--real_age", type=int, help="Real age of the person in the image")
parser.add_argument("--dataset", type=int, help="dataset used")

model_path = parser.parse_args().model_path
image_path = parser.parse_args().image_path
real_age = parser.parse_args().real_age
dataset = parser.parse_args().dataset

print('-----------------')
print('best model path:',model_path)
print('image path:',image_path)
print('-----------------')

# Ruta al modelo entrenado y a la imagen

# Cargar el modelo y la imagen
RANDOM_SEED = 1
GRAYSCALE = False

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
if dataset == 'cacd':
    model = init_resnet(GRAYSCALE)
else:
    model = init_resnet50_sense(GRAYSCALE)

model.load_state_dict(torch.load(model_path))
# Cargar los pesos del mejor modelo
#load_best_model_weights(model,model_path)
image_tensor = preprocess_image(image_path)

# Predecir la edad
predicted_age = predict_age(model, image_tensor)
print(predicted_age)
print(f"Predicted Age: {int(predicted_age + 16)} years")
print(f"Real age is: {real_age} years")
