# coding: utf-8

#############################################
# Cross Entropy with ResNet-34
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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
import wandb

torch.backends.cudnn.deterministic = True
print(os.getcwd(),'\n\n\n')
#TRAIN_CSV_PATH = '/dataset/afad_train.csv'
#TEST_CSV_PATH = '/home/xnmaster/projecte/XNAPproject-grup_15/dataset/afad_test.csv'
#IMAGE_PATH = '/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/AFAD/AFAD-Full'

TRAIN_CSV_PATH = './dataset_split/RANGE_splitted_datasets/afad_splitRANGE_train.csv'
TEST_CSV_PATH = './dataset_split/RANGE_splitted_datasets/afad_splitRANGE_test.csv'
IMAGE_PATH = 'datasets_img/AFAD/AFAD-Full'
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
NUM_CLASSES = 10 # canviat per fer afad amb dataset RANGE 0-9
BATCH_SIZE = 256
GRAYSCALE = False

wandb.init(
    # set the wandb project where this run will be logged
        project="projecte-deep",
        
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "ce",
            "dataset": "afad-resnet18",
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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, grayscale):
        self.inplanes = 64
        if grayscale:
            in_dim = 1
        else:
            in_dim = 3
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model

def resnet18(num_classes, grayscale):
    """Constructs a ResNet-18 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[2,2,2,2],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


class FireModule(nn.Module):
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(FireModule, self).__init__()
        self.squeeze = nn.Conv2d(in_channels=in_channels, out_channels=s1x1, kernel_size=1, stride=1)
        self.expand1x1 = nn.Conv2d(in_channels=s1x1, out_channels=e1x1, kernel_size=1)
        self.expand3x3 = nn.Conv2d(in_channels=s1x1, out_channels=e3x3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.squeeze(x))
        x1 = self.expand1x1(x)
        x2 = self.expand3x3(x)
        x = F.relu(torch.cat((x1, x2), dim=1))
        return x
    

class SqueezeNet(nn.Module):
    def __init__(self, out_channels):
        super(SqueezeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=7, stride=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fire2 = FireModule(in_channels=96, s1x1=16, e1x1=64, e3x3=64)
        self.fire3 = FireModule(in_channels=128, s1x1=16, e1x1=64, e3x3=64)
        self.fire4 = FireModule(in_channels=128, s1x1=32, e1x1=128, e3x3=128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fire5 = FireModule(in_channels=256, s1x1=32, e1x1=128, e3x3=128)
        self.fire6 = FireModule(in_channels=256, s1x1=48, e1x1=192, e3x3=192)
        self.fire7 = FireModule(in_channels=384, s1x1=48, e1x1=192, e3x3=192)
        self.fire8 = FireModule(in_channels=384, s1x1=64, e1x1=256, e3x3=256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.fire9 = FireModule(in_channels=512, s1x1=64, e1x1=256, e3x3=256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=1, stride=1)
        self.avg_pool = nn.AvgPool2d(kernel_size=12, stride=1)
        # We don't have FC Layers, inspired by NiN architecture.
        
    def forward(self, x):
        # First max pool after conv1
        x = self.max_pool1(self.conv1(x))
        # Second max pool after fire4
        x = self.max_pool2(self.fire4(self.fire3(self.fire2(x))))
        # Third max pool after fire8
        x = self.max_pool3(self.fire8(self.fire7(self.fire6(self.fire5(x)))))
        # Final pool (avg in this case) after conv10
        x = self.avg_pool(self.conv10(self.fire9(x)))
        return torch.flatten(x, start_dim=1)
    
def Squeezenet(num_classes):
    """Constructs a SqueezeNet model."""
    model = SqueezeNet(num_classes)
    return model  
    
###########################################
# Initialize Cost, Model, and Optimizer
###########################################

torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = resnet18(NUM_CLASSES, GRAYSCALE)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 


def compute_mae_and_mse(model, data_loader, device):
    mae, mse, num_examples = 0., 0., 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        mae += torch.sum(torch.abs(predicted_labels - targets))
        mse += torch.sum((predicted_labels - targets)**2)
    
    mae = mae.float()/num_examples
    mse = mse.float()/num_examples
       
    return mae, mse


start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets.to(DEVICE)

        # FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        # UPDATE MODEL PARAMETERS
        optimizer.step()

        # LOGGING
        if not batch_idx % 50:
            
            s = ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f'
                 % (epoch+1, num_epochs, batch_idx,
                     len(train_dataset)//BATCH_SIZE, cost))

        # Registra las métricas en W&B
            wandb.log({'Epoch': epoch+1, 'Batch': batch_idx, 'Train Loss': cost.item()})
            print(s)
            with open(LOGFILE, 'a') as f:
                f.write('%s\n' % s)


    model.eval()
    with torch.set_grad_enabled(False):
        test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                                device=DEVICE)
        
        train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE)
        
        wandb.log({'Train MAE': train_mae.item(), 'Train RMSE': torch.sqrt(train_mse).item(),
               'Test MAE': test_mae.item(), 'Test RMSE': torch.sqrt(test_mse).item(), 
               'epoch': epoch})

    if test_mae < best_mae:
        best_mae, best_rmse, best_epoch = test_mae, torch.sqrt(test_mse), epoch
        ########## SAVE MODEL #############
        torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


    s = 'MAE/RMSE: | Current Test: %.2f/%.2f Ep. %d | Best Test : %.2f/%.2f Ep. %d' % (
        test_mae, torch.sqrt(test_mse), epoch, best_mae, best_rmse, best_epoch)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

model.eval()
with torch.set_grad_enabled(False):  # save memory during inference

    train_mae, train_mse = compute_mae_and_mse(model, train_loader,
                                               device=DEVICE)
    test_mae, test_mse = compute_mae_and_mse(model, test_loader,
                                             device=DEVICE)

    # Registra las métricas en Weights & Biases
    wandb.log({'Train MAE B': train_mae.item(), 'Train RMSE B': torch.sqrt(train_mse).item(),
               'Test MAE B': test_mae.item(), 'Test RMSE B': torch.sqrt(test_mse).item()})

    s = 'MAE/RMSE: | Train: %.2f/%.2f | Test: %.2f/%.2f' % (
        train_mae, torch.sqrt(train_mse), test_mae, torch.sqrt(test_mse))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
print(s)
with open(LOGFILE, 'a') as f:
    f.write('%s\n' % s)

s = 'Best MAE: %.2f | Best RMSE: %.2f | Best Epoch: %d' % (best_mae, best_rmse, best_epoch)
print(s)
with open(LOGFILE, 'a') as f:
    f.write('%s\n' % s)

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
