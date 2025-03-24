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

from config import Config
# from Experiment_celeba_Attack.celeba.data_celeba_1000 import CelebA
from resnet import resnet18
from model import Crop, Face_Inversion, DeNormalize
from skip import *
from model_stylegan2 import Generator, Discriminator

import matplotlib.pyplot as plt

# import nevergrad as ng

from learning_PSP_inversion_forDefense_noise import GradualStyleEncoder

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

    init = Init_Func(config.init_func)
    for name, W in model.named_parameters():
        if 'conv' in name and 'bias' not in name:
            W.data = init.init(W.data)

    config.model = model.to(device)

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

    inv_model = Face_Inversion(config.model.cpu(), config.layers)
    config.model = config.model.to(device)
    inv_model = inv_model.to(device)
    inv_model.eval()

    latent = 512
    n_mlp = 8
    g_ema = Generator(
    64, latent, n_mlp, channel_multiplier=1).to(device)
    checkpoint = torch.load("./models/070000.pt")
    g_ema.load_state_dict(checkpoint["g_ema"])

    encoder = GradualStyleEncoder(50, 'ir_se', config.layers[0])
    config.encoder = encoder.to(device)
    config.encoder.load_state_dict(torch.load('./PSP_checkpoints/PSP_encoder_inversion_addavg_layer2_foradv_3.pt'.format(str(config.layers[0])), map_location=device)['encoder'])
    config.encoder.eval()

    print("encoder:")
    print(encoder)

    
    inverseIter = iter(test_loader)
    for i in range(config.inverse_num):
        img_true, _ = next(inverseIter)
        img1 = config.denorm(img_true.detach())
        vutils.save_image(img1, 'recon_pics/{}/ref_{}_Gan.png'.format(n, i), normalize=False)
        img_true = img_true.to(device)
        
        with torch.no_grad():
            config.model(img_true)
            intermedia_feature = inv_model.hook.get_feat()
            
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

        intermedia_feature = intermedia_feature.to(device)
        inversion_result = intermedia_feature.detach().clone()
        latent_avg = latent_mean.detach().clone().unsqueeze(0).to(device)
        latent_in = config.encoder(inversion_result).detach().clone() #+ latent_avg
        latent_in.requires_grad = True

        optimizer = optim.Adam(params = [latent_in], lr = config.lr)
        step = 1000
        noise_level = 0.04
        noise_ramp = 0.75
        latent_path = []

        for i1 in range(config.iter_w):
            t = i1 / step
            noise_strength = latent_std * noise_level * max(0, 1 - t / noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            sample, _ = g_ema([latent_n + latent_avg], input_is_latent=True, truncation=1) # latent code 不加噪声效果不好，找不到比较好的近似
      
            Min=-1
            Max=1
            sample.clamp_(min=Min, max=Max)
            sample.add_(-Min).div_(Max - Min + 1e-5)
                
            sample = norm(sample)
                
            config.model(sample)
            XFeature = inv_model.hook.get_feat()
            
            featureLoss = ((intermedia_feature - XFeature)**2).mean()
                
            TVLoss = TV(sample)
            normLoss = l2loss(sample)
                
            optimizer.zero_grad()
            totalLoss = featureLoss# + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss
            totalLoss.backward(retain_graph=True)
            optimizer.step()

            if (i1) % 99 == 0:
                img = config.denorm(sample.cpu().detach())
                ssim_ = ssim(img, img1)
                lpips_loss = LPIPS() 
                lpips_  =lpips_loss(img, img1)
                psnr_ = psnr(img, img1)
                vutils.save_image(img, 'recon_pics/{}/{}/recon_{}_{}_Gan_ssim_{:.4}_psnr_{:.4}_lpips_{:.4}.png'.format(n, d, i, i1, ssim_, psnr_, lpips_), normalize=False)
                
                
            print ("Iter optimize w+ ", i1, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy())
            writer.add_scalars('w+1', {'Feature loss': featureLoss}, i1)
            writer.add_scalars('w+2', {'TV Loss': TVLoss}, i1)
            writer.add_scalars('w+3', {'l2 Loss': normLoss}, i1)
            
        img = config.denorm(sample.cpu().detach())
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
