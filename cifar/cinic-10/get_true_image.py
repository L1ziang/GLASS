import argparse
import os
import shutil
from tkinter.ttk import Style
from xml.parsers.expat import model
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from piq import ssim, psnr, LPIPS

import random
import numpy as np

from config import Config
from resnet import resnet18
from model import DeNormalize
from skip import *

import matplotlib.pyplot as plt

import nevergrad as ng
import torchvision

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy
from torch_utils import gen_utils

import io

# import pyiqa
# import cv2

# from nrvqa.niqe import niqe

import sys
sys.path.append('..')
from utils import *

SEED = 666
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
cudnn.deterministic = True
cudnn.benchmark = False

os.makedirs('get_image1/', exist_ok=True)
cinic_directory = '../data/cinic'
cinic_mean = [0.4749, 0.4691, 0.4255]
cinic_std = [0.2406, 0.2366, 0.2574]



train_dataset = torchvision.datasets.CIFAR10('./data', train=True, download=True,transform=transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean,std=cinic_std)]),)
cinic_test = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=1, shuffle=True)

denorm = DeNormalize(mean=cinic_mean, std=cinic_std)

inverseIter = iter(cinic_test)
for i in range(1200):
    img_true, label_true = inverseIter.next()
    img1 = denorm(img_true.detach())
    vutils.save_image(img1, 'get_image1/{}.png'.format(i), normalize=False)