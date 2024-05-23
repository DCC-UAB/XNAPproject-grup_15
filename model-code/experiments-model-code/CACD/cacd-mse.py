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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import transforms
from PIL import Image
import wandb

torch.backends.cudnn.deterministic = True
"""
TRAIN_CSV_PATH = './datasets/cacd_train.csv'
VALID_CSV_PATH = './datasets/cacd_valid.csv'
TEST_CSV_PATH = './cacd_test.csv'
IMAGE_PATH = '/shared_datasets/CACD/centercropped/jpg'
"""
TRAIN_CSV_PATH = r"/home/xnmaster/projecte/XNAPproject-grup_15/datasets/cacd-train1.csv"

TEST_CSV_PATH = r"/home/xnmaster/projecte/XNAPproject-grup_15/datasets/cacd-test1.csv"
IMAGE_PATH = r'/home/xnmaster/projecte/XNAPproject-grup_15/datasets_img/CACD/CACD2000'
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

parser.add_argument('--imp_weight',
                    type=int,
                    default=0)

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

IMP_WEIGHT = args.imp_weight

PATH = args.outpath
if not os.path.exists(PATH):
    os.mkdir(PATH)
LOGFILE = os.path.join(PATH, 'training.log')
TEST_PREDICTIONS = os.path.join(PATH, 'test_predictions.log')
TEST_ALLPROBAS = os.path.join(PATH, 'test_allprobas.tensor')

# Logging

header = []

header.append('PyTorch Version: %s' % torch.__version__)
header.append('CUDA device available: %s' % torch.cuda.is_available())
header.append('Using CUDA device: %s' % DEVICE)
header.append('Random Seed: %s' % RANDOM_SEED)
header.append('Task Importance Weight: %s' % IMP_WEIGHT)
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
NUM_CLASSES = 49
BATCH_SIZE = 256
GRAYSCALE = False

df = pd.read_csv(TRAIN_CSV_PATH, index_col=0)
ages = df['age'].values
del df
ages = torch.tensor(ages, dtype=torch.float)



# Wandb initialization
wandb.init(
    # set the wandb project where this run will be logged
        entity='xisca',
        project="projecte-deep",
        name = 'cacd ordinal ',
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": "MSE",
            "model": "mse",
            "dataset": "cacd",
            "epochs": num_epochs,
            }
    )

def task_importance_weights(label_array):
    uniq = torch.unique(label_array)
    num_examples = label_array.size(0)

    m = torch.zeros(uniq.shape[0])

    for i, t in enumerate(torch.arange(torch.min(uniq), torch.max(uniq))):
        m_k = torch.max(torch.tensor([label_array[label_array > t].size(0), 
                                      num_examples - label_array[label_array > t].size(0)]))
        m[i] = torch.sqrt(m_k.float())

    imp = m/torch.max(m)
    return imp


# Data-specific scheme
if not IMP_WEIGHT:
    imp = torch.ones(NUM_CLASSES-1, dtype=torch.float)
elif IMP_WEIGHT == 1:
    imp = task_importance_weights(ages)
    imp = imp[0:NUM_CLASSES-1]
else:
    raise ValueError('Incorrect importance weight parameter.')
imp = imp.to(DEVICE)


###################
# Dataset
###################

class CACDDataset(Dataset):
    """Custom Dataset for loading CACD face images"""

    def __init__(self,
                 csv_path, img_dir, transform=None):

        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df['file'].values
        self.y = df['age'].values
        self.transform = transform

    def __getitem__(self, index):
       
        imatge = os.path.join(self.img_dir,
                                    self.img_names[index])
        img = Image.open(imatge)
    
        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        levels = [1]*label + [0]*(NUM_CLASSES - 1 - label)
        levels = torch.tensor(levels, dtype=torch.float32)

        return img, label, levels

    def __len__(self):
        return self.y.shape[0]


custom_transform = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomCrop((120, 120)),
                                       transforms.ToTensor()])

train_dataset = CACDDataset(csv_path=TRAIN_CSV_PATH,
                            img_dir=IMAGE_PATH,
                            transform=custom_transform)


custom_transform2 = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.CenterCrop((120, 120)),
                                       transforms.ToTensor()])

test_dataset = CACDDataset(csv_path=TEST_CSV_PATH,
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
        self.num_classes = num_classes
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
        self.fc = nn.Linear(512, (self.num_classes-1)*2)

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
        logits = logits.view(-1, (self.num_classes-1), 2)
        probas = F.softmax(logits, dim=2)[:, :, 1]
        return logits


def resnet34(num_classes, grayscale):
    """Constructs a ResNet-34 model."""
    model = ResNet(block=BasicBlock, 
                   layers=[3, 4, 6, 3],
                   num_classes=num_classes,
                   grayscale=grayscale)
    return model


###########################################
# Initialize Cost, Model, and Optimizer
###########################################

def cost_fn(logits, levels, imp):
    val = (-torch.sum((F.log_softmax(logits, dim=2)[:, :, 1]*levels
                      + F.log_softmax(logits, dim=2)[:, :, 0]*(1-levels))*imp, dim=1))
    return torch.mean(val)


torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
model = resnet34(NUM_CLASSES, GRAYSCALE)

model.to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

# Métricas
def compute_mae_mse_and_r2(model, data_loader, device):
    mae_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    mae, mse, r2, num_examples = 0., 0., 0., 0
    for i, (features, targets, l) in enumerate(data_loader):
        features = features.to(device)
        targets = targets.to(device).float()

        logits = model(features)
        logits= logits[:, 0, 0]
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


start_time = time.time()

best_mae, best_rmse, best_epoch = 999, 999, -1
for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, targets, levels) in enumerate(train_loader):

        features = features.to(DEVICE)
        targets = targets
        targets = targets.to(DEVICE).float()
        levels = levels.to(DEVICE) 


        # FORWARD AND BACK PROP
        logits = model(features)
        
        cost = cost_fn(logits, levels, imp)
        logits_p = logits
        logits= logits[:, 0, 0] # nose pq thas de carregar aquestes dimensions
        cost = F.mse_loss(logits, targets)
        optimizer.zero_grad()

        cost.backward()

        # UPDATE MODEL PARAMETERS
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
    probas = F.softmax(logits_p, dim=2)[:, :, 1]

    with torch.set_grad_enabled(False):
        train_mae, train_mse , train_r2= compute_mae_mse_and_r2(model, train_loader,
                                               device=DEVICE)

        test_mae, test_mse, test_r2 = compute_mae_mse_and_r2(model, test_loader,
                                             device=DEVICE)
        wandb.log({
            'epoch': epoch, 
            'train_mae': train_mae, 'train_mse': train_mse, 'train_r2': train_r2, 
            'test_mae': test_mae, 'test_mse': test_mse, 'test_r2': test_r2
        })
    if test_mae < best_mae:
        best_mae, best_rmse, best_epoch = test_mae, (test_mse)**0.5, epoch
        ########## SAVE MODEL #############
        torch.save(model.state_dict(), os.path.join(PATH, 'best_model.pt'))


    s = 'MAE/RMSE: | Current Test: %.2f/%.2f Ep. %d | Best Test : %.2f/%.2f Ep. %d' % (
        test_mae, (test_mse)**0.5, epoch, best_mae, best_rmse, best_epoch)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

    s = 'Time elapsed: %.2f min' % ((time.time() - start_time)/60)
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

model.eval()
with torch.set_grad_enabled(False):  # save memory during inference

    train_mae, train_mse, train_r2 = compute_mae_mse_and_r2(model, train_loader,
                                               device=DEVICE)

    test_mae, test_mse, test_r2 = compute_mae_mse_and_r2(model, test_loader,
                                             device=DEVICE)

    s = 'MAE/RMSE: | Train: %.2f/%.2f | Test: %.2f/%.2f' % (
        train_mae, 0.5**(train_mse),
      
        test_mae, 0.5**(test_mse))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)

s = 'Total Training Time: %.2f min' % ((time.time() - start_time)/60)
print(s)
with open(LOGFILE, 'a') as f:
    f.write('%s\n' % s)


########## EVALUATE BEST MODEL ######
model.load_state_dict(torch.load(os.path.join(PATH, 'best_model.pt')))
model.eval()

with torch.set_grad_enabled(False):
    train_mae, train_mse ,train_r2 = compute_mae_mse_and_r2(model, train_loader,
                                               device=DEVICE)
    test_mae, test_mse, test_r2 = compute_mae_mse_and_r2(model, test_loader,
                                             device=DEVICE)

    s = 'MAE/RMSE: | Best Train: %.2f/%.2f | Best Test: %.2f/%.2f' % (
        train_mae, 0.5**(train_mse),
    
        test_mae, 0.5**(test_mse))
    print(s)
    with open(LOGFILE, 'a') as f:
        f.write('%s\n' % s)


########## SAVE PREDICTIONS ######
all_pred = []
all_probas = []
with torch.set_grad_enabled(False):
    for batch_idx, (features, targets, levels) in enumerate(test_loader):

        features = features.to(DEVICE)
        logits= model(features)
        all_probas.append(probas)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)

        lst = [str(int(i)) for i in predicted_labels]
        all_pred.extend(predicted_labels)

torch.save(torch.cat(all_probas).to(torch.device('cpu')), TEST_ALLPROBAS)
with open(TEST_PREDICTIONS, 'w') as f:
    all_pred = ','.join(all_pred)
    f.write(all_pred)

    
    