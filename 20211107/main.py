from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/resnet18')

from tqdm import tqdm_notebook
import numpy as np
from datetime import datetime
import argparse 

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary as summary_
import torchvision
import torchvision.transforms as transforms

import data_loader
import utils
import calibration
import train as t

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--modelname', default='resnet18', type=str, help='modelname')
args = parser.parse_args()

trainloader, validloader = data_loader.get_train_valid_loader(data_dir='./data',
                           batch_size=128,
                           augment=True,
                           random_seed=42,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=False)

modelname = args.modelname
epochs = 500 # referred the paper "On Calibration of Modern Neural Networks" Figure 3
today_ymd = utils.today_ymd

model = torch.hub.load('pytorch/vision:v0.10.0', modelname, pretrained=True)
model.fc = nn.Linear(512, len(utils.label_names))
model = model.cuda()
model = model.train()
trunk = nn.DataParallel(model)

calib = calibration.CalibrationLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(trunk.parameters(), lr=.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

ex = t.Experiments(writer, modelname, today_ymd)
ex.load_data(trainloader, validloader)
ex.load_model(trunk)
ex.load_prerequisite(criterion, optimizer, scheduler)
ex.run()