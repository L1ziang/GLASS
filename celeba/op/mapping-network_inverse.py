
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
from piq import ssim, psnr
import pandas as pd
from sklearn.decomposition import PCA
import sys
sys.path.append('..')
sys.path.append('../..')
# from config import Config
# from Experiment_celeba_Attack.celeba.data_celeba_1000 import CelebA
from resnet import resnet18
from model import Crop, Face_Inversion, DeNormalize
from skip import *
from model_stylegan2 import Generator, Discriminator

import matplotlib.pyplot as plt

import nevergrad as ng
import math

# import sys
# sys.path.append('..')
from utils import *

from torch.utils.cpp_extension import load
module_path = os.path.dirname(__file__)

fused = load(
    "fused",
    sources=[
        os.path.join(module_path, "fused_bias_act.cpp"),
        os.path.join(module_path, "fused_bias_act_kernel.cu"),
    ],
)

from torch.autograd import Function

def fused_leaky_relu(input, bias=None, negative_slope=0.2, scale=2 ** 0.5):
    if bias is not None:
        rest_dim = [1] * (input.ndim - bias.ndim - 1)
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=5.0
            )
        )


class EqualLinear_inverse(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=0.01, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        # self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul

        # self.inverse_leaky_relu = nn.LeakyReLU(negative_slope=5)
        


    def forward(self, input):
        if self.activation:
            # out = self.inverse_leaky_relu(input+self.bias)
            out = fused_leaky_relu(input, self.bias**0.000442, negative_slope=5.0)
            out = F.linear(input, self.weight*0.000442)

        return out

class mapping_inverse_network(nn.Module):
    def __init__(self):
        super(mapping_inverse_network, self).__init__()

        layers = []
        for i in range(8):
            layers.append(
                EqualLinear_inverse(
                    # 512, 512, lr_mul=0.01, activation="fused_lrelu"
                    512, 512, activation="fused_lrelu"
                )
            )
        self.bn = nn.BatchNorm1d(512)
        layers.append(self.bn)

        self.style = nn.Sequential(*layers)

    def forward(self, input):
        out = self.style(input)
        return out

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def l2loss(x):
    return (x**2).mean()

def main():

    use_cuda = torch.cuda.is_available()
    device = torch.device(0 if use_cuda else 'cpu')

    # 加载生成模型
    latent = 512
    n_mlp = 8
    g_ema = Generator(
    64, latent, n_mlp, channel_multiplier=1).to(device)
    checkpoint = torch.load("../models/070000_mlp8.pt")
    g_ema.load_state_dict(checkpoint["g_ema"])
    
    
    # 这里要初始化restarts个sample_z，分别优化iter_z轮后，选择featureLoss最小的z，来进行后面w的优化
    # SEED = np.random.randint(10000) # 随机数种子来自np，随后固定并打印
    SEED = 6666
    # SEED = 6361

    np.random.seed(SEED)
    print("The np SEED is ", SEED)
    
    # restarts = config.restarts
    # sample_z = [None for _ in range(restarts)]
    # optimizer = [None for _ in range(restarts)]
    # scores = torch.zeros(restarts)
    # sample_z_numpy = np.random.randn(restarts, latent)
    # sample_z = torch.from_numpy(sample_z_numpy, requires_grad = True, device=device) # (restarts, latent)
    # optimizer = []

    g_ema.eval()
    g_ema = g_ema.to(device)

    # for p in g_ema.style[:1]:
    Lenth = len(g_ema.style)
    # for i in range(Lenth-1, 0, -1):
    #     p = g_ema.style[i]
    #     print(p.weight.shape) # torch.Size([512, 512])
    #     print(p.bias.shape) # torch.Size([512])
    #     print(p.scale) # 0.00044194173824159215
    #     print(p.lr_mul) #0.01
    # exit()

    t = torch.randn(1, 512, device=device)
    print(t.shape)
    n = 0
    # for p in g_ema.style[1:]:
    #     t = p(t)
    #     n = n+1
    #     if n == 2:
    #         print(t.shape) # torch.Size([1, 512])
    #         print(t.ndim) # 2
    #         print(p.bias.ndim) # 1
    #         exit()

    X = torch.randn(1, 512, device=device) # 相当于X
    X_pixelnorm = X * torch.rsqrt(torch.mean(X ** 2, dim=1, keepdim=True) + 1e-8)
    Y = g_ema.style(X)

    for i in range(Lenth-1, 0, -1):
        # print(i)
        p = g_ema.style[i]
        # print(p.lr_mul) # 0.01
        # exit()
        Y = Y / (2 ** 0.5)
        empty = Y.new_empty(0)
        Y = fused.fused_bias_act(Y.contiguous(), (p.bias*p.lr_mul), empty, 3, 0, 5.0, (2 ** 0.5))
        # Y = F.leaky_relu(Y, negative_slope=5.)
        Y = Y - (p.bias*p.lr_mul).view(1, p.bias.shape[0])
        # Y = Y / p.scale
        # Y = F.linear(Y, (p.weight*p.scale).inverse())
        Y = Y.matmul((p.weight*p.scale ).t().inverse())
        # if i == 6:
            # print(Y)
            # exit()
    
    print(abs(Y - X_pixelnorm).mean())



    

    exit()



    # with torch.no_grad(): # 求w的mean和std n_mean_latent设为10000
    #     n_mean_latent = 20000 # 构造2w组数据
    #     noise_sample = torch.randn(n_mean_latent, 512, device=device) # 相当于X
    #     latent_out = g_ema.style(noise_sample) # 相当于Y

    #     latent_mean = latent_out.mean(0)
    #     latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
    #     # 统计Y的均值方差

    net = mapping_inverse_network()
    net = net.to(device)
    net.train()

    EPOCH = 2000
    # noise_sample.to(device)
    # latent_out.to(device)

    requires_grad(net, True)
    # latent_out.requires_grad_()

    optimizer= optim.Adam(params = net.parameters(), lr = 0.01)
    # print(g_ema.style)
    # exit()

    # 优化方法:
    # with torch.no_grad(): # 求w的mean和std n_mean_latent设为10000
    #     n_mean_latent = 1 # 构造2w组数据
    #     noise_sample = torch.randn(n_mean_latent, 512, device=device) # 相当于X
    #     latent_out = g_ema.style(noise_sample) # 相当于Y
    
    # O = torch.randn(1, 512, device=device) # 相当于X
    # O.requires_grad_()
    # noise_sample.to(device)
    # latent_out.to(device)
    # O.to(device)

    # optimizer = optim.Adam(params = [O], lr = 0.1)

    # for i in range(EPOCH):

    #     # O_mean = O.detach().clone().mean(0)
    #     # O_std = ((O.detach().clone() - O_mean).pow(2).sum() / 1) ** 0.5 +0.1
    #     # predict = g_ema.style((O-O_mean)/O_std)
    #     predict = g_ema.style(O)

    #     optimizer.zero_grad()

    #     # loss = (abs(predict - latent_out)).mean()
    #     # mean_loss = O.mean()
    #     # std_loss = ((O - O.mean(0)).pow(2).sum())** 0.5
    #     KLD = -0.5 * torch.sum(1 + torch.log(torch.std(O.squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(O.squeeze(), axis=-1).pow(2) - torch.std(O.squeeze(), unbiased=False, axis=-1).pow(2))
    #     # loss = ((predict - latent_out)**2).mean() + 2*mean_loss + 2*(1 - std_loss)
    #     l2 = l2loss(O)
    #     loss = ((predict - latent_out)**2).mean() +0.5*KLD +5*l2
    #     L = ((O - noise_sample)**2).mean()

    #     loss.backward()
    #     optimizer.step()

    #     print("iter : {}  loss : {}  KLDloss : {} l2loss : {} L : {}".format(i, loss.item(), KLD.item(), l2.item(), L.item()))

    # print(abs(O-noise_sample))
    # print(noise_sample)


    #     # latent_out.requires_grad_()

    #     # noise_sample_predicted = net(latent_out)
    
    # exit()


    # 映射方法:
    for i in range(EPOCH):

        with torch.no_grad(): # 求w的mean和std n_mean_latent设为10000
            n_mean_latent = 2000 # 构造2w组数据
            noise_sample = torch.randn(n_mean_latent, 512, device=device) # 相当于X
            latent_out = g_ema.style(noise_sample) # 相当于Y

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
            # 统计Y的均值方差

        noise_sample.to(device)
        latent_out.to(device)

        latent_out.requires_grad_()

        noise_sample_predicted = net(latent_out)
        optimizer.zero_grad()
        # loss = ((noise_sample_predicted - noise_sample)**2).mean()
        loss = (abs(noise_sample_predicted - noise_sample)).mean()
        loss.backward()
        optimizer.step()

        print("iter : {}   loss : {}".format(i, loss.item()))






if __name__ == '__main__':
    main()