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

from config import Config
# from Experiment_celeba_Attack.celeba.data_celeba_1000 import CelebA
from resnet import resnet18
from model import Crop, Face_Inversion, DeNormalize
from skip import *
from model_stylegan2 import Generator, Discriminator

import matplotlib.pyplot as plt

# import nevergrad as ng

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
            target = target.to(device)
            target = target[:,2].view(-1,1).float() #Attractive
            output = model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            _out = (torch.sigmoid(output)>0.5)
            correct += _out.eq(target.view_as(_out)).sum().item()

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

def create_gen_model():
    gen_model = skip(3, 3, num_channels_down = [16, 32, 64, 128, 128],
                                      num_channels_up =   [16, 32, 64, 128, 128],
                                      num_channels_skip = [4, 4, 4, 4, 4],
                                      filter_size_down = [7, 7, 5, 5, 3],  # type: ignore
                                      filter_size_up = [7, 7, 5, 5, 3],   # type: ignore
                      upsample_mode='nearest', downsample_mode='avg', need_sigmoid=True, pad='zero', act_fun='LeakyReLU') #.type(torch.cuda.FloatTensor)
    gen_model.cuda()
    return gen_model

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise

def main():
    parser = argparse.ArgumentParser(description="optimize_Gan")
    parser.add_argument('--config_file', type=str, default='', help="config file")
    parser.add_argument('--stage', type=str, default='', help="select the training stage")
    parser.add_argument('--index', type=str, default='', help="the index of the expriments")
    args = parser.parse_args()

    config = Config(args)

    torch.manual_seed(config.random_seed)

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
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': False}

    test_dataset = torch.load("./data/celeba_top3000ID_64_test.npz")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **cuda_kwargs)

    print("len(test_data):", len(test_dataset))

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

    config.denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    
    
    inverseIter = iter(test_loader)
    for i in range(config.inverse_num):
        img_true, _ = next(inverseIter)
        img1 = config.denorm(img_true.detach()) # 保存原图
        vutils.save_image(img1, 'recon_pics/{}/ref_{}_Gan.png'.format(n, i), normalize=False)
        img_true = img_true.to(device)
        
        with torch.no_grad():
            intermedia_feature = config.model.getLayerOutput(img_true, config.layers) #torch.Size([1, 128, 8, 8])
            
        latent = 512
        n_mlp = 8
        g_ema = Generator(
        64, latent, n_mlp, channel_multiplier=1).to(device)
        checkpoint = torch.load("./models/070000.pt")
        g_ema.load_state_dict(checkpoint["g_ema"])
        
        SEED = np.random.randint(10000) 
        np.random.seed(SEED)
        print("The np SEED is ", SEED)

        g_ema.eval()
        g_ema = g_ema.to(device)

        with torch.no_grad():
            n_mean_latent = 10000
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)

            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        restarts = config.restarts
        sample_z = [None for _ in range(restarts)]
        optimizer = [None for _ in range(restarts)]
        scores = torch.zeros(restarts)
        
        for re in range(restarts): 
            
            sample_z_numpy = np.random.randn(1, latent) # (1, 512)
            sample_z[re] = torch.from_numpy(sample_z_numpy).type(torch.float32).to(device)
            sample_z[re].requires_grad_()

            if config.optimizer =='adam': 
                optimizer[re] = optim.Adam(params = [sample_z[re]], lr = 1e-2, eps = 1e-3, amsgrad = config.AMSGrad)
            
        g_ema.eval() 
        for z_index in range(restarts):
            for i1 in range(config.iter_z):
                sample, _ = g_ema([sample_z[z_index]], truncation=1)
                Min=-1
                Max=1
                sample.clamp_(min=Min, max=Max)
                sample.add_(-Min).div_(Max - Min + 1e-5)
                
                sample = norm(sample)
                
                XFeature = config.model.getLayerOutput(sample, config.layers)
                
                optimizer[z_index].zero_grad()
                featureLoss = ((intermedia_feature - XFeature)**2).mean()
                
                TVLoss = TV(sample)
                normLoss = l2loss(sample)

                KLD = -0.5 * torch.sum(1 + torch.log(torch.std(sample_z[z_index].squeeze(), unbiased=False, axis=-1).pow(2) + 1e-10) - torch.mean(sample_z[z_index].squeeze(), axis=-1).pow(2) - torch.std(sample_z[z_index].squeeze(), unbiased=False, axis=-1).pow(2))
                totalLoss = featureLoss + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss + config.lambda_KLD * KLD #+ z_loss*2
                
                totalLoss = featureLoss + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss + config.lambda_KLD * KLD #+ z_loss*2
                totalLoss.backward(retain_graph=True)
                optimizer[z_index].step()
                
                
                print ("z_index ", z_index, "Iter optimize z ", i1, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy(), "KLD: ", KLD.cpu().detach().numpy())
                writer.add_scalars('z1', {'Feature loss': featureLoss}, i1)
                writer.add_scalars('z2', {'TV Loss': TVLoss}, i1)
                writer.add_scalars('z3', {'l2 Loss': normLoss}, i1)

            scores[z_index] = featureLoss.detach() # scores取detach
             
        tok1_of_restarts_indices = torch.topk(scores, k=1, largest=False)
        latent_init_z = sample_z[tok1_of_restarts_indices.indices[0]].data
        latent_init_w = g_ema.style(latent_init_z)
        latent_in = latent_init_w.detach().clone()#.unsqueeze(0).repeat(1, 1)
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        latent_in.requires_grad = True
            
        
        optimizer = optim.Adam(params = [latent_in], lr = config.lr, eps = 1e-3, amsgrad = config.AMSGrad)
        step = 10000
        noise_level = 0.02
        noise_ramp = 0.75
        latent_path = []

        for i2 in range(config.iter_w):
            t = i2 / step
            noise_strength = latent_std * noise_level * max(0, 1 - t / noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            sample, _ = g_ema([latent_n], input_is_latent=True, truncation=1)
            Min=-1
            Max=1
            sample.clamp_(min=Min, max=Max)
            sample.add_(-Min).div_(Max - Min + 1e-5)
                
            sample = norm(sample)
            
            XFeature = config.model.getLayerOutput(sample, config.layers)

            featureLoss = ((intermedia_feature - XFeature)**2).mean()
                
            TVLoss = TV(sample)
            normLoss = l2loss(sample)
                
            optimizer.zero_grad()
            totalLoss = featureLoss + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss
            totalLoss.backward(retain_graph=True)
            optimizer.step()

            if (i2 + 1) % 100 == 0: 
                img = config.denorm(sample.cpu().detach()) 
                vutils.save_image(img, 'recon_pics/{}/{}/recon_{}_{}_Gan.png'.format(n, d, i, i2), normalize=False)
                
                
            print ("Iter optimize w+ ", i2, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy())
            writer.add_scalars('w+1', {'Feature loss': featureLoss}, i2)
            writer.add_scalars('w+2', {'TV Loss': TVLoss}, i2)
            writer.add_scalars('w+3', {'l2 Loss': normLoss}, i2)
            
        img = config.denorm(sample.cpu().detach())
        vutils.save_image(img, 'recon_pics/{}/recon_{}_Gan.png'.format(n, i), normalize=False)
        
            
    writer.close()
    print("=> Training Complete!\n")



if __name__ == '__main__':
    main()
