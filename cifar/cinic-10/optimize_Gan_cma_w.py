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

import sys
sys.path.append('..')
from utils import *

best_test_acc = 0
best_inv_loss = 0
def save_checkpoint(config, test_acc, inv_loss):
    global best_test_acc, best_inv_loss
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        save_path = os.path.join(config.model_dir, "best_acc_"+config.save_model)
        print ('\nnew best acc is {}'.format(best_test_acc))
        print ('saving model {}'.format(save_path))
        state = {
            'model': config.model.state_dict(),
            'best_test_acc': best_test_acc
        }
        torch.save(state, save_path)
    if inv_loss > best_inv_loss:
        best_inv_loss = inv_loss
        save_path = os.path.join(config.model_dir, "best_loss_"+config.save_model)
        print ('\nnew best inv loss is {}'.format(best_inv_loss))
        print ('saving model {}'.format(save_path))
        state = {
            'model': config.model.state_dict(),
            'best_inv_loss': best_inv_loss
        }
        torch.save(state, save_path)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)

def _tensor_size(t):
    return t.size()[1]*t.size()[2]*t.size()[3]

def TV(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:,:,1:,:])
    count_w = _tensor_size(x[:,:,:,1:])
    h_tv = torch.pow(x[:,:,1:,:]-x[:,:,:h_x-1,:], 2).sum()
    w_tv = torch.pow(x[:,:,:,1:]-x[:,:,:,:w_x-1], 2).sum()
    return (h_tv / count_h + w_tv / count_w) / batch_size

def l2loss(x):
    return (x**2).mean()

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def main():
    truncation_psi = 1
    centroids_path = None
    translate = 0.0
    rotate = 0

    parser = argparse.ArgumentParser(description="optimize_Gan")
    parser.add_argument('--config_file', type=str, default='', help="config file")
    parser.add_argument('--stage', type=str, default='', help="select the training stage")
    parser.add_argument('--index', type=str, default='', help="the index of the expriments")
    parser.add_argument('--seed', default=666, type=int)
    args = parser.parse_args()
    config = Config(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

    os.makedirs(config.model_dir, exist_ok=True)
    n = config.layers[0] + '_' + args.index
    d = 'details'
    os.makedirs('recon_pics/{}'.format(n), exist_ok=True)
    os.makedirs('recon_pics/{}/{}'.format(n, d), exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device(config.gpu if use_cuda else 'cpu')

    cuda_kwargs = {}
    if use_cuda:
        torch.cuda.set_device(config.gpu)
        cudnn.benchmark = True
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': True}

    cinic_directory = '../data/cinic'
    cinic_mean = [0.4749, 0.4691, 0.4255]
    cinic_std = [0.2406, 0.2366, 0.2574]

    cinic_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder('./attacked_images',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
        batch_size=1, shuffle=True)

    print("len(test_data):", len(cinic_test))

    model = None
    if config.arch == 'face_cnn':
        model = resnet18(pretrained=False)
    else:
        raise Exception("unknown model architecture")
    
    print("\ntarget model:", model, "\n")

    ### for initialization experiments
    init = Init_Func(config.init_func)
    for name, W in model.named_parameters():
        if 'conv' in name and 'bias' not in name:
            W.data = init.init(W.data)

    config.model = model.to(device)

    # unlike resume, load model does not care optimizer status or start_epoch
    if config.load_model:
        model_path = os.path.join(config.model_dir, config.load_model)
        print('==> Loading from {}'.format(model_path))
        config.model.load_state_dict(torch.load(model_path, map_location=device)['model'])
        
        
    config.model = model.to(device)
    config.model.eval()

    name = args.stage + args.index
    writer = SummaryWriter(os.path.join(config.log_dir, name))

    config.denorm = DeNormalize(mean=cinic_mean, std=cinic_std)
    norm = transforms.Normalize(mean=cinic_mean, std=cinic_std)
    
    # 测模型准确率的
    # loss, acc = test(config.model, device, test_loader)
    # print("acc : ", acc)
    
    inverseIter = iter(cinic_test)
    for i in range(config.inverse_num):
        img_true, label_true = inverseIter.next()
        img1 = config.denorm(img_true.detach()) 
        vutils.save_image(img1, 'recon_pics/{}/ref_{}_Gan.png'.format(n, i), normalize=False)
        img_true = img_true.to(device)
        label_true = label_true.to(device)
        
        with torch.no_grad():
            intermedia_feature = config.model.getLayerOutput(img_true, config.layers) #torch.Size([1, 128, 8, 8])
            
            
        with open('./pretrained_models/cifar10.pkl', "rb") as f:
            buffer = io.BytesIO(f.read())
            G = legacy.load_network_pkl(buffer)['G_ema']
            G = G.eval().requires_grad_(False).to(device)

        
        SEED = np.random.randint(10000)
        np.random.seed(SEED)
        print("The np SEED is ", SEED)

        restarts = config.restarts
        sample_z = [None for _ in range(restarts)]
        optimizer = [None for _ in range(restarts)]
        scores = torch.zeros(restarts)
        
        for re in range(restarts): 
            sample_z_numpy = np.random.randn(1, G.z_dim)
            sample_z[re] = torch.from_numpy(sample_z_numpy).type(torch.float32).to(device)
            sample_z[re].requires_grad_()
            if config.optimizer =='adam': 
                optimizer[re] = optim.Adam(params = [sample_z[re]], lr = 1e-2, eps = 1e-3, amsgrad = config.AMSGrad)
            
        w_avg = G.mapping.w_avg[label_true.item()].unsqueeze(0).repeat(1, 1)
        w_avg = w_avg.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)
        class_indices = torch.full((1,), label_true.item()).to(device)
        
        for z_index in range(restarts):
            for i1 in range(1):

                label = F.one_hot(label_true, G.c_dim)
                w = G.mapping(sample_z[z_index], label)
                w = w_avg + (w - w_avg) * truncation_psi

                sample = gen_utils.w_to_tensor(G, w, to_np=True)

                Min=-1
                Max=1
                sample.clamp_(min=Min, max=Max)
                sample.add_(-Min).div_(Max - Min + 1e-5)

                sample = norm(sample)

                XFeature = config.model.getLayerOutput(sample, config.layers)
                
                featureLoss = ((intermedia_feature - XFeature)**2).mean()
                
                TVLoss = TV(sample)
                normLoss = l2loss(sample)
                
                KLD = -0.5 * torch.sum(1 + torch.log(torch.std(sample_z[z_index].squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(sample_z[z_index].squeeze(), axis=-1).pow(2) - torch.std(sample_z[z_index].squeeze(), unbiased=False, axis=-1).pow(2))
                totalLoss = featureLoss + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss #+ config.lambda_KLD * KLD #+ z_loss*2
                
                totalLoss = featureLoss + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss + config.lambda_KLD * KLD #+ z_loss*2
                
                print ("z_index ", z_index, "Iter optimize z ", i1, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy(), "KLD: ", KLD.cpu().detach().numpy())
                writer.add_scalars('z1', {'Feature loss': featureLoss}, i1)
                writer.add_scalars('z2', {'TV Loss': TVLoss}, i1)
                writer.add_scalars('z3', {'l2 Loss': normLoss}, i1)

            scores[z_index] = featureLoss.detach() 
             
        tok1_of_restarts_indices = torch.topk(scores, k=1, largest=False)
        latent_init_z = sample_z[tok1_of_restarts_indices.indices[0]].data
        w = G.mapping(latent_init_z, label)
        latent_init_w = w_avg + (w - w_avg) * truncation_psi
        latent_in = latent_init_w[0][0].unsqueeze(0).detach().clone()

        budget = 20000
        strategy='CMA'
        L = latent_in.detach().cpu().numpy()
        parametrization = ng.p.Array(init=L).set_mutation(sigma=0.4)#.set_bounds(lower=-128, upper=128)#.set_mutation(sigma=1)#.set_bounds(lower=-64, upper=64)#.set_mutation(sigma=0.8)#.set_mutation(sigma=1.0)
        optimizer = ng.optimizers.registry[strategy](parametrization=parametrization, budget=budget)

        step = 10000
        noise_level = 0.01 #0.05
        noise_ramp = 0.75
        latent_path = []

        for i2 in range(config.iter_w):
            l1 = optimizer.ask()
            l2 = torch.Tensor(l1.value).type(torch.float32).to(device) # (1, 512)
            latent_n = l2.unsqueeze(1).repeat(1, G.mapping.num_ws, 1)

            latent_n = w_avg + (latent_n- w_avg) * truncation_psi
            sample = gen_utils.w_to_tensor(G, latent_n, to_np=True)
            Min=-1
            Max=1
            sample.clamp_(min=Min, max=Max)
            sample.add_(-Min).div_(Max - Min + 1e-5)
            sample = norm(sample)
            XFeature = config.model.getLayerOutput(sample, config.layers)

            featureLoss = ((intermedia_feature - XFeature)**2).mean()
                
            TVLoss = TV(sample)
            normLoss = l2loss(sample)
                
            totalLoss = featureLoss + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss
            optimizer.tell(l1, totalLoss.item())

            if (i2 + 1) % 100 == 0 or i2 == 0:
                img = config.denorm(sample.cpu().detach()) 

                ssim_ = ssim(img, img1)
                lpips_loss = LPIPS() 
                lpips_  =lpips_loss(img, img1)
                psnr_ = psnr(img, img1)
                vutils.save_image(img, 'recon_pics/{}/{}/recon_{}_{}_Gan_ssim_{:.4}_psnr_{:.4}_lpips_{:.4}.png'.format(n, d, i, i2, ssim_, psnr_, lpips_), normalize=False)
                
            print ("Iter optimize w+ ", i2, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy())
            writer.add_scalars('w+1', {'Feature loss': featureLoss}, i2)
            writer.add_scalars('w+2', {'TV Loss': TVLoss}, i2)
            writer.add_scalars('w+3', {'l2 Loss': normLoss}, i2)
            
        img = config.denorm(sample.cpu().detach()) # 保存原图
        # SSIM img, img1
        ssim_ = ssim(img, img1)
        lpips_loss = LPIPS() 
        lpips_  =lpips_loss(img, img1)
        psnr_ = psnr(img, img1)
        vutils.save_image(img, 'recon_pics/{}/recon_{}_Gan_ssim_{:.4}_psnr_{:.4}_lpips_{:.4}.png'.format(n, i, ssim_, psnr_, lpips_), normalize=False)

    writer.close()
    print("=> Training Complete!\n")

if __name__ == '__main__':
    main()
