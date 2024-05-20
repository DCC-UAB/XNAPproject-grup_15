import os
import shutil
from PIL import Image
import torch
import torchvision.transforms as transforms
import pandas as pd

# DATA AUGMENTATION
DA_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomCrop((120, 120)),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.RandomResizedCrop(size=(120, 120), scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.05),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
])

# Rutes de las carpetas
source_folder = "/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/AFAD/AFAD-Full"
destination_folder = "/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/AFAD/AFAD-Full-DA"

# Copiem estructura FAD Full
shutil.copytree(source_folder, destination_folder)

# Llista per desar info pel csv
images_info = []

# Iterar sobre les carpetas y subcarpetas
for root, dirs, files in os.walk(destination_folder):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            # Generem info csv
            folder_name = os.path.basename(root)
            age = int(folder_name) - 15  # Restar 15 para obtener la edad real
            gender = "male" if "111" in root else "female"
            
            # Generar ruta
            full_file_path = os.path.join(root, file)
            
            # Obtenir imagen
            img = Image.open(full_file_path)
            
            # Aplicar transformacions
            img_tensor = DA_transforms(img)
            
            # Convertir el tensor a una img PIL
            img_transformed = transforms.ToPILImage()(img_tensor)
            
            # Guardar la imagen transformada
            img_transformed.save(full_file_path)
            
            # Agregar informació a la llista
            images_info.append({"file": file, "path": full_file_path, "age": age, "gender": gender})
            
            # Imprimir un msg cada 100 imatges procesades
            if len(images_info) % 100 == 0:
                print(f"{len(images_info)} imágenes procesadas.")

# Crear DataFrame i desar
df = pd.DataFrame(images_info)
df.to_csv("/home/xnmaster/projecte/XNAPproject-grup_15/datasets/afad-propi-DA-train.csv", index=False)

print('!!!Done!!!')

