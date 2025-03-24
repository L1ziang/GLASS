import argparse
import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from piq import ssim, psnr

from config import Config
from resnet import resnet18
from resnet import encoder_network
from model import Crop, Face_Inversion, DeNormalize
from model_stylegan2 import Generator, Discriminator
from model_stylegan2 import EqualLinear

import sys
sys.path.append('..')
from utils import *
import os

import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module
from helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE
import numpy as np

from criteria.lpips.lpips import LPIPS
from criteria import w_norm

class GradualStyleBlock(Module):
    def __init__(self, in_c, out_c, spatial):
        super(GradualStyleBlock, self).__init__()
        self.out_c = out_c
        self.spatial = spatial
        num_pools = int(np.log2(spatial))
        modules = []
        modules += [Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1),
                    nn.LeakyReLU()]
        for i in range(num_pools - 1):
            modules += [
                Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU()
            ]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.out_c)
        x = self.linear(x)
        return x
    
def make_layers(in_channels, config):
    layers = []

    for out_channels in config:
        upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
        layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
        in_channels = out_channels
    upconv2d = nn.ConvTranspose2d(in_channels, 3, 4, 2, 1)
    layers += [upconv2d, nn.Tanh()]

    return nn.Sequential(*layers)

class GradualStyleEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(GradualStyleEncoder, self).__init__()
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        decoder_config = [128, 64]
        decoder = make_layers(128, decoder_config)

        self.input_layer = Sequential(decoder,
                                      Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)

        self.styles = nn.ModuleList()
        self.style_count = 10 # opts.n_styles
        self.coarse_ind = 3
        self.middle_ind = 7
        for i in range(self.style_count):
            if i < self.coarse_ind:
                style = GradualStyleBlock(512, 512, 8) # Without reducing the number of convolutional layers, training would collapse before the first epoch1000 sets of data
                # style = GradualStyleBlock(512, 512, 16)
            elif i < self.middle_ind:
                style = GradualStyleBlock(512, 512, 8)
                # style = GradualStyleBlock(512, 512, 32)
            else:
                style = GradualStyleBlock(512, 512, 16)
                # style = GradualStyleBlock(512, 512, 64)
            self.styles.append(style)
        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True) + y

    def forward(self, x):
        x = self.input_layer(x)

        latents = []
        modulelist = list(self.body._modules.values())
        for i, l in enumerate(modulelist):
            x = l(x)
            if i == 8:
                c1 = x
            elif i == 14:
                c2 = x
            elif i == 17:
                c3 = x

        for j in range(self.coarse_ind):
            latents.append(self.styles[j](c3))

        p2 = self._upsample_add(c3, self.latlayer1(c2))
        for j in range(self.coarse_ind, self.middle_ind):
            latents.append(self.styles[j](p2))

        p1 = self._upsample_add(p2, self.latlayer2(c1))
        for j in range(self.middle_ind, self.style_count):
            latents.append(self.styles[j](p1))

        out = torch.stack(latents, dim=1)
        return out



def inv_train(config, device, train_iter, optimizer, epoch, len_loader):
    config.model.eval()
    config.inv_model.train()

    batch_idx, (data, target) = next(train_iter)
    data = data.to(device)

    optimizer.zero_grad()

    with torch.no_grad():
        config.model(data)
        feature = config.inv_model.hook.get_feature(config.inv_model.layer_id)

    inv_output = config.inv_model(feature)
    inv_loss = F.mse_loss(inv_output, data)
    inv_loss.backward()

    optimizer.step()

    if batch_idx % config.print_freq == 0:
        print("InvTrain Epoch: {} [{:.0f}%]\tLoss: {:.4f}".format(
              epoch, 100. * batch_idx / len_loader, inv_loss.item()))

    return batch_idx


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


def learning_gan_test(config, device, test_loader, epoch, n, encoder, discriminator, g_ema):
    config.model.eval()
    encoder.eval()
    discriminator.eval()
    config.inv_model.eval()
    correct = 0
    test_loss = 0
    inv_loss = 0
    plot = config.plot

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device)
            target = target[:,2].view(-1,1).float()

            output = config.model(data)

            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            _out = (torch.sigmoid(output)>0.5).float()
            correct += _out.eq(target.view_as(_out)).sum().item()

            feature = config.inv_model.hook.get_feature(config.inv_model.layer_id)
            w = encoder(feature)
            x_rec, _ = g_ema([w], truncation=1)
            
            inv_loss += F.mse_loss(x_rec, data, reduction='sum').item() # sum up batch inv loss

            if plot:
                truth = data[0:32]
                inverse = x_rec[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                out = config.denorm(out.detach())
                vutils.save_image(out, 'recon_pics/{}/recon_{}.png'.format(n,epoch), normalize=False)
                plot = False

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)            
    inv_loss /= len(test_loader.dataset) * 3 * 64 * 64

    if config.adv:
        save_checkpoint(config, test_acc, inv_loss)

    print('\nTest set: test_acc: {:.2f}%, test_loss: {:.4f}, inv_loss: {:.4f}\n'
          .format(test_acc, test_loss, inv_loss))

    return test_acc, test_loss, inv_loss

def test(config, device, test_loader, epoch, n):
    config.model.eval()
    config.inv_model.eval()
    correct = 0
    test_loss = 0
    inv_loss = 0
    plot = config.plot

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device)
            target = target[:,2].view(-1,1).float()

            output = config.model(data)

            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            _out = (torch.sigmoid(output)>0.5).float()
            correct += _out.eq(target.view_as(_out)).sum().item()

            feature = config.inv_model.hook.get_feature(config.inv_model.layer_id)
            inv_output = config.inv_model(feature)
            inv_loss += F.mse_loss(inv_output, data, reduction='sum').item() # sum up batch inv loss

            if plot:
                truth = data[0:32]
                inverse = inv_output[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                out = config.denorm(out.detach())
                vutils.save_image(out, 'recon_pics/{}/recon_{}.png'.format(n,epoch), normalize=False)
                plot = False

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)            
    inv_loss /= len(test_loader.dataset) * 3 * 64 * 64

    if config.adv:
        save_checkpoint(config, test_acc, inv_loss)

    print('\nTest set: test_acc: {:.2f}%, test_loss: {:.4f}, inv_loss: {:.4f}\n'
          .format(test_acc, test_loss, inv_loss))

    return test_acc, test_loss, inv_loss

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()



def main():
    parser = argparse.ArgumentParser(description="Adversarial Training Against Model Inversion Attack Example")
    parser.add_argument('--config_file', type=str, default='', help="config file")
    parser.add_argument('--stage', type=str, default='', help="select the training stage")
    parser.add_argument('--index', type=str, default='', help="the index of the expriments")

    args = parser.parse_args()
    config = Config(args)

    torch.manual_seed(config.random_seed)

    os.makedirs(config.model_dir, exist_ok=True)
    n = config.layers[0] + '_' + args.index
    if config.inv:
        os.makedirs('recon_pics/{}'.format(n), exist_ok=True)
        
    use_cuda = torch.cuda.is_available()
    device = torch.device(config.gpu if use_cuda else 'cpu')

    cuda_kwargs = {}
    if use_cuda:
        torch.cuda.set_device(config.gpu)
        cudnn.benchmark = True
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': False}

    model = None
    if config.arch == 'face_cnn':
        model = resnet18(pretrained=False)
    else:
        raise Exception("unknown model architecture")
    
    encoder = GradualStyleEncoder(50, 'ir_se')

    latent = 512
    n_mlp = 8
    g_ema = Generator(
    64, latent, n_mlp, channel_multiplier=1).to(device)
    checkpoint = torch.load("./models/070000.pt", map_location=device)
    g_ema.load_state_dict(checkpoint["g_ema"])
    
    print("\ntarget model:", model, "\n")

    init = Init_Func(config.init_func)
    for name, W in encoder.named_parameters():
        if 'conv' in name and 'bias' not in name:
            W.data = init.init(W.data)

    config.model = model.to(device)
    encoder = encoder.to(device)


    # unlike resume, load model does not care optimizer status or start_epoch
    if config.load_model:
        model_path = os.path.join(config.model_dir, config.load_model)
        print('==> Loading from {}'.format(model_path))
        config.model.load_state_dict(torch.load(model_path, map_location=device)['model'])
        config.model.eval()
    inv_model = Face_Inversion(config.model.cpu(), config.layers)
    config.model = config.model.to(device) # 暂时没用
    config.denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    with torch.no_grad():
        n_mean_latent = 10000
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    encoder_optimizer = optim.Adam(encoder.parameters(), config.lr, eps = 1e-3, amsgrad=True)

    start_from_latent_avg = True

    lpips_loss = LPIPS(net_type='alex').to(device).eval()
    w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=start_from_latent_avg)

    inv_model = inv_model.to(device)
    inv_model.eval()

    test_data = torch.load("./data/celeba_top3000ID_64_test.npz")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False, **cuda_kwargs)

    attack_test_data = Attack_Dataset()
    model = model.to(device)
    for data, _ in test_loader:
        data = data.to(device)
        with torch.no_grad():
            model(data)
            feature = inv_model.hook.get_feat()
        feature = feature.to('cpu')
        data = data.to('cpu')
        attack_test_data.push(feature, data)

    attack_test_loader = torch.utils.data.DataLoader(attack_test_data, batch_size=128, shuffle=False, **cuda_kwargs)

    train_data = torch.load("./data/celeba_other3000ID_64.npz")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=60, shuffle=True, **cuda_kwargs)

    encoder.train()
    g_ema.eval()
    start_epoch = 0

    sample_test_iter = iter(attack_test_loader)
    sample_test, sample_test_O = next(sample_test_iter)
    sample_test = sample_test.to(device)
    log = 1
    loss_min = 999999999.0

    for epoch in range(start_epoch, config.epochs+1):
        encoder.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data_O = data.to(device)
            with torch.no_grad():
                config.model(data_O)
                feature = inv_model.hook.get_feat()
            data_R = feature.to(device)

            codes = encoder(data_R)
            if start_from_latent_avg:
                latent_avg = latent_mean.detach().clone().unsqueeze(0).to(device)
                codes = codes + latent_avg#.repeat(codes.shape[0], 1)
                
            data_rec, _ = g_ema([codes], input_is_latent=True, truncation=1)

            loss_l2 = F.mse_loss(data_rec, data_O)
            loss_lpips = lpips_loss(data_rec, data_O)
            loss_w_norm = w_norm_loss(codes, latent_avg)

            l2_lambda = 1
            lpips_lambda = 0.5
            w_norm_lambda = 0.001
            Loss = l2_lambda * loss_l2 + lpips_lambda * loss_lpips + w_norm_lambda * loss_w_norm #+ ssim_lambda * loss_ssim#+ weight_l2_lambda * loss_weight_l2

            if log >= 2000 and Loss > 0.4: # 0.4 for stem, 0.2 for others
                print("Loss is too large.")
            else:
                encoder_optimizer.zero_grad()
                Loss.backward()
                encoder_optimizer.step()
                print('epoch {} : Loss: {} l2 : {} lpips : {} w_norm {}'.format(epoch, Loss, loss_l2, loss_lpips, loss_w_norm))

            if log%200 == 0:
                A = data_O.detach().clone()
                B = data_rec.detach().clone()
                out = torch.cat((A, B))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = A[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = B[i * 8:i * 8 + 8]
                out = config.denorm(out.detach())
                vutils.save_image(out, 'recon_pics/{}/log_{}.png'.format(n,log), normalize=False)
            log = log + 1


        encoder.eval()
        with torch.no_grad():
            code_test = encoder(sample_test) # torch.Size([32, 3, 64, 64])
            if start_from_latent_avg:
                latent_avg = latent_mean.detach().clone().unsqueeze(0).to(device)
                code_test = code_test + latent_avg#.repeat(codes.shape[0], 1)
            data_rec_test, _ = g_ema([code_test], input_is_latent=True, truncation=1)
            A = sample_test_O.detach().clone().to(device)
            B = data_rec_test.detach().clone()
            out = torch.cat((A, B))
            for i in range(4):
                out[i * 16:i * 16 + 8] = A[i * 8:i * 8 + 8]
                out[i * 16 + 8:i * 16 + 16] = B[i * 8:i * 8 + 8]
            out = config.denorm(out.detach())
            vutils.save_image(out, 'recon_pics/{}/recon_{}.png'.format(n,epoch), normalize=False)

        # save the encoder
        with torch.no_grad():
            loss_sum = 0
            for batch_idx, (data_R, data_O) in enumerate(attack_test_loader):
                data_R = data_R.to(device)
                data_O = data_O.to(device)
                codes = encoder(data_R)
                if start_from_latent_avg:
                    latent_avg = latent_mean.detach().clone().unsqueeze(0).to(device)
                    codes = codes + latent_avg#.repeat(codes.shape[0], 1)
                    
                data_rec, _ = g_ema([codes], input_is_latent=True, truncation=1)

                loss_l2 = F.mse_loss(data_rec, data_O)
                loss_lpips = lpips_loss(data_rec, data_O)
                loss_w_norm = w_norm_loss(codes, latent_avg)
                l2_lambda = 1
                lpips_lambda = 0.6 #0.3
                w_norm_lambda = 0.001
                Loss = l2_lambda * loss_l2 + lpips_lambda * loss_lpips + w_norm_lambda * loss_w_norm #+ weight_l2_lambda * loss_weight_l2

                loss_sum += Loss
            loss_sum /= len(test_loader.dataset)
            print("Epoch: {}, The sum of loss is {}".format(epoch, loss_sum))
            if loss_sum < loss_min:
                loss_min = loss_sum
                print("Get the minimal loss, save the encoder.")
                torch.save(
                    {
                        "encoder": encoder.state_dict(),
                    },
                    f"./PSP_checkpoints/PSP_encoder_inversion_addavg_{str(config.layers[0])}.pt",
                )
            

    exit()



if __name__ == '__main__':
    # encoder = GradualStyleEncoder(50, 'ir_se')
    # print(encoder)
    # x = torch.Tensor(1,128,8,8)
    # y = encoder(x)
    # exit()
    # CUDA_VISIBLE_DEVICES=1     
    main()
