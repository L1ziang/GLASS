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
from piq import ssim, psnr, LPIPS

from config import Config
# from data import CelebA
from resnet import resnet18
from model import Crop, DeNormalize, Face_Inversion, CELEBA_Dropout_Classifier, Face_Inversion_noise

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

def test(config, device, test_loader, epoch, n):
    config.model.eval()
    config.inv_model.eval()
    correct = 0
    test_loss = 0
    inv_loss = 0
    plot = config.plot
    ssim_v = 0
    psnr_v = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = target[:,2].view(-1,1).float()

            output = config.model(data)

            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            _out = (torch.sigmoid(output)>0.5).float()
            correct += _out.eq(target.view_as(_out)).sum().item()

            feature = config.inv_model.hook.get_feat()
            inv_output = config.inv_model(feature)
            inv_loss += F.mse_loss(inv_output, data, reduction='sum').item() # sum up batch inv loss

            inv_output = config.denorm(inv_output.detach())
            data = config.denorm(data.detach())
            ssim_v += ssim(inv_output, data, reduction='sum').item()
            psnr_v += psnr(inv_output, data, reduction='sum').item()

            if plot:
                truth = data[0:32]
                inverse = inv_output[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                # out = config.denorm(out.detach())
                vutils.save_image(out, 'recon_pics/{}/recon_{}.png'.format(n,epoch), normalize=False)
                plot = False

    inv_loss /= len(test_loader.dataset) * 3 * 128 * 128
    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)  
    ssim_v /= len(test_loader.dataset)
    psnr_v /= len(test_loader.dataset)

    if config.adv:
        save_checkpoint(config, test_acc, inv_loss)

    print("\nTest set: test_acc: {:.2f}%, test_loss: {:.4f}, "
          "inv_loss: {:.4f}, ssim_v: {:.4f}, psnr_v: {:.4f}\n"
          .format(test_acc, test_loss, inv_loss, ssim_v, psnr_v))

    return test_acc, test_loss, inv_loss

def noisy_train(config, device, train_loader, optimizer, epoch, criterion):
    config.model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target[:,2].view(-1,1).float()

        optimizer.zero_grad()

        output, _ = config.model(data)
        # loss = criterion(data, inter_feat, output, target)
        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()

        optimizer.step()

        if batch_idx % config.print_freq == 0:
            print("NoisyTrain Epoch: {} [{:.0f}%]\tLoss: {:.4f}"
                  .format(epoch, 100. * batch_idx / len(train_loader), loss.item()))

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

def adv_train(config, device, train_iter, optimizer, epoch, len_loader):
    config.model.train()
    config.inv_model.eval()

    batch_idx, (data, target) = next(train_iter)
    data = data.to(device)
    target = target.to(device)
    target = target[:,2].view(-1,1).float()

    optimizer.zero_grad()

    output = config.model(data)
    feature = config.inv_model.hook.get_feature(config.inv_model.layer_id)
    inv_output = config.inv_model(feature)

    ce_loss = F.binary_cross_entropy_with_logits(output, target)
    inv_loss = F.mse_loss(inv_output, data)
    total_loss = ce_loss - (config.gamma * inv_loss)

    total_loss.backward()

    optimizer.step()

    if batch_idx % config.print_freq == 0:
        print("AdvTrain Epoch: {} [{:.0f}%]\t"
              "Loss: {:.4f}\t"
              "ce_loss: {:.6f}\t"
              "inv_loss: {:.6f}\n"
              .format(epoch, 100. * batch_idx / len_loader, 
              total_loss.item(), ce_loss.item(), inv_loss.item()))

def adv_train_O(config, device, train_iter, optimizer, epoch, len_loader, noise):
    config.model.train()
    config.inv_model.eval()

    for param in config.model.parameters():
        param.requires_grad = False

    for param in config.model.conv1.parameters():
        param.requires_grad = True
    for param in config.model.bn1.parameters():
        param.requires_grad = True
    config.model.relu.requires_grad = True
    config.model.maxpool.requires_grad = True
    for param in config.model.layer1.parameters():
        param.requires_grad = True
    for param in config.model.layer2.parameters():
        param.requires_grad = True

    batch_idx, (data, target) = next(train_iter)
    data = data.to(device)
    target = target.to(device)
    target = target[:,2].view(-1,1).float()

    optimizer.zero_grad()

    output = config.model(data)
    feature = config.inv_model.hook.get_feature(config.inv_model.layer_id)
    inv_output = config.inv_model(feature)

    ce_loss = F.binary_cross_entropy_with_logits(output, target)
    inv_loss = 1 - ssim(config.denorm(inv_output), config.denorm(data))
    noise = noise.repeat(data.size(0), 1, 1, 1)
    noise_loss = 1 - ssim(noise, config.denorm(data))
    total_loss = ce_loss + (config.gamma * (noise_loss - inv_loss))

    total_loss.backward()

    optimizer.step()

    if batch_idx % config.print_freq == 0:
        print("AdvTrain Epoch O: {} [{:.0f}%]\t"
              "Loss: {:.4f}\t"
              "ce_loss: {:.6f}\t"
              "inv_loss: {:.6f}\n"
              .format(epoch, 100. * batch_idx / len_loader, 
              total_loss.item(), ce_loss.item(), inv_loss.item()))
        
def adv_train_C(config, device, train_iter, optimizer, epoch, len_loader):
    config.model.train()
    config.inv_model.eval()

    for param in config.model.parameters():
        param.requires_grad = True
    for param in config.model.conv1.parameters():
        param.requires_grad = False
    for param in config.model.bn1.parameters():
        param.requires_grad = False
    config.model.relu.requires_grad = False
    config.model.maxpool.requires_grad = False
    for param in config.model.layer1.parameters():
        param.requires_grad = False
    for param in config.model.layer2.parameters():
        param.requires_grad = False

    batch_idx, (data, target) = next(train_iter)
    data = data.to(device)
    target = target.to(device)
    target = target[:,2].view(-1,1).float()

    optimizer.zero_grad()

    output = config.model(data)
    feature = config.inv_model.hook.get_feature(config.inv_model.layer_id)
    inv_output = config.inv_model(feature)

    ce_loss = F.binary_cross_entropy_with_logits(output, target)
    inv_loss = F.mse_loss(inv_output, data)
    total_loss = ce_loss #- (config.gamma * inv_loss)

    total_loss.backward()

    optimizer.step()

    if batch_idx % config.print_freq == 0:
        print("AdvTrain Epoch C: {} [{:.0f}%]\t"
              "Loss: {:.4f}\t"
              "ce_loss: {:.6f}\t"
              "inv_loss: {:.6f}\n"
              .format(epoch, 100. * batch_idx / len_loader, 
              total_loss.item(), ce_loss.item(), inv_loss.item()))

def noisy_test(config, device, test_loader, criterion, n):
    config.model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = target[:,2].view(-1,1).float()

            output, _ = config.model(data)
            test_loss += F.binary_cross_entropy_with_logits(output, target, reduction='sum').item()  # sum up batch loss
            _out = (torch.sigmoid(output)>0.5).float()
            correct += _out.eq(target.view_as(_out)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)

    print('\nTest set: test_acc: {:.2f}%, test_loss: {:.4f}\n'
          .format(test_acc, test_loss))

    return test_acc, test_loss


def inv_test(config, device, test_loader, epoch, n):
    config.inv_model.eval()
    inv_loss = 0
    ssim_v = 0
    psnr_v = 0
    plot = config.plot

    with torch.no_grad():
        for feat, target in test_loader:
            feat = feat.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            inv_output = config.inv_model(feat)
            inv_loss += F.mse_loss(inv_output, target, reduction='sum').item()

            inv_output = config.denorm(inv_output.detach())
            target = config.denorm(target.detach())
            ssim_v += ssim(inv_output, target, reduction='sum').item()
            psnr_v += psnr(inv_output, target, reduction='sum').item()

            if plot:
                truth = target[0:32]
                inverse = inv_output[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, 'recon_pics/{}/recon_{}.png'.format(n,epoch), normalize=False)
                plot = False

    inv_loss /= len(test_loader.dataset) * 3 * 128 * 128
    ssim_v /= len(test_loader.dataset)
    psnr_v /= len(test_loader.dataset)
    print('\nTest set: inv_loss: {:.4f}\tssim_v: {:.4f}\tpsnr_v: {:.4f}\n'.format(inv_loss, ssim_v, psnr_v))

    return inv_loss, ssim_v, psnr_v


best_test_acc = 0
best_inv_loss = 999


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--config_file', type=str, default='', help="config file")
    parser.add_argument('--stage', type=str, default='', help="select the training stage")
    parser.add_argument('--index', type=str, default='', help="the index of the expriments")
    args = parser.parse_args()

    global best_test_acc, best_inv_loss
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
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': True}

    train_data = torch.load("./data/celeba_top3000ID_64_train.npz")
    test_data = torch.load("./data/celeba_top3000ID_64_test.npz")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, **cuda_kwargs)
    train_loader1 = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, **cuda_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False, **cuda_kwargs)

    inv_data = torch.load("./data/celeba_other3000ID_64.npz")
    inv_train_loader = torch.utils.data.DataLoader(inv_data, batch_size=256, shuffle=True, **cuda_kwargs)

    print("len(adv_data):", len(train_data))
    print("len(inv_data):", len(inv_data))
    print("len(test_data):", len(test_data))



    model = None
    if config.arch == 'face_cnn':
        model = resnet18(pretrained=False)
    else:
        raise Exception("unknown model architecture")
    inv_model = Face_Inversion(model, config.layers)

    print("\ntarget model:", model, "\n")

    init = Init_Func(config.init_func)
    for name, W in model.named_parameters():
        if 'conv' in name and 'bias' not in name:
            W.data = init.init(W.data)

    config.model = model.to(device)
    config.inv_model = inv_model.to(device)

    if config.load_model:
        model_path = os.path.join(config.model_dir, config.load_model)
        print('==> Loading from {}'.format(model_path))
        config.model.load_state_dict(torch.load(model_path, map_location=device)['model'])
    if config.load_inv_model:
        inv_model_path = os.path.join(config.model_dir, config.load_inv_model)
        print('==> Loading from {}'.format(inv_model_path))
        config.inv_model.load_state_dict(torch.load(inv_model_path, map_location=device)['inv_model'])

    optimizer = None
    inv_optimizer = None
    scheduler = None
    if config.adv:
        if config.optimizer == 'sgd':
            print("adv model using SGD optimizer")
            optimizer = optim.SGD(config.model.parameters(),config.lr,momentum=0.9,weight_decay=1e-4)
        elif config.optimizer =='adam':
            print("adv model using ADAM optimizer")
            optimizer = optim.Adam(config.model.parameters(), config.lr)

        if config.lr_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs,eta_min=4e-08)
        elif config.lr_scheduler == 'step':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(config.epochs/2), int(config.epochs/4*3)])

        print("inv model using ADAM optimizer")
        inv_optimizer = optim.Adam(config.inv_model.parameters(), config.inv_lr)

    elif config.inv:
        print("inv model using ADAM optimizer")
        inv_optimizer = optim.Adam(config.inv_model.parameters(), config.lr)

    start_epoch = 1
    if config.resume:
        if os.path.isfile(config.resume):
            checkpoint = torch.load(config.resume)
            start_epoch = checkpoint['epoch']
            config.model.load_state_dict(checkpoint['model'])
            config.inv_model.load_state_dict(checkpoint['inv_model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            inv_optimizer.load_state_dict(checkpoint['inv_optimizer'])
            print("=> loaded checkpoint '{}' (acc {}, epoch {})"
                  .format(config.resume, checkpoint['best_acc'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config.resume))

    save_path = os.path.join(config.model_dir, config.save_model)
    name = args.stage + args.index
    writer = SummaryWriter(os.path.join(config.log_dir, name))
    config.denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    noise_shape = (3, 64, 64)
    noise = torch.randn(noise_shape) 
    noise = (noise - torch.min(noise)) / (torch.max(noise) - torch.min(noise))
    noise = noise.to(device)   

    best_recon_loss = 999
    for epoch in range(start_epoch, config.epochs+1):

        len_train = len(train_loader)
        len_inv_train = len(inv_train_loader)
        train_enum = enumerate(train_loader)
        inv_train_enum = enumerate(inv_train_loader)

        if config.adv:
            for i in range(len_train):
                if i % 3 == 0:
                    adv_train_O(config, device, train_enum, optimizer, epoch, len_train, noise)
                if i % 3 == 1:
                    inv_train(config, device, train_enum, inv_optimizer, epoch, len_train)
                if i % 3 == 2:
                    adv_train_C(config, device, train_enum, optimizer, epoch, len_train)

            test_acc, test_loss, inv_loss = test(config, device, test_loader, epoch, n)
            writer.add_scalars('loss', {'test_loss': test_loss, 'inv_loss': inv_loss}, epoch)
            writer.add_scalars('accuracy', {'test_acc': test_acc}, epoch)
            
            if epoch % config.ckpt_interval == 0 and epoch != (config.epochs+1):
                ckpt_path = os.path.join(config.model_dir, config.save_model.replace('.pth', '_ckpt.pth'))
                print ('saving ckpt {}\n'.format(ckpt_path))
                state = {
                    'epoch': epoch,
                    'test_acc': test_acc,
                    'inv_loss': inv_loss,
                    'model': config.model.state_dict(),
                    'inv_model': config.inv_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'inv_optimizer': inv_optimizer.state_dict(),
                }
                torch.save(state, ckpt_path)

        elif config.inv:
            for i in range(len_inv_train):
                inv_train(config, device, inv_train_enum, inv_optimizer, epoch, len_inv_train)

            _, _, inv_loss = test(config, device, test_loader, epoch, n)
            writer.add_scalars('loss', {'inv_loss': inv_loss}, epoch)

            if inv_loss < best_recon_loss:
                best_recon_loss = inv_loss
                state = {
                    'epoch': epoch,
                    'inv_model': config.inv_model.state_dict(),
                    'optimizer': inv_optimizer.state_dict(),
                    'best_inv_loss': best_recon_loss
                }
                print ("new best inv loss is {}, saving best inv model\n".format(best_recon_loss))
                torch.save(state, save_path)
                shutil.copyfile('recon_pics/{}/recon_{}.png'.format(n,epoch), 'recon_pics/{}/best_test.png'.format(n))

        if scheduler:
            scheduler.step()
        else:
            pass # it uses adam


    if config.adv:
        state = {
            'model': config.model.state_dict(),
        }
        print('saving the final model {}'.format(save_path))
        torch.save(state, save_path)

    writer.close()
    print("=> Training Complete!\n")



if __name__ == '__main__':
    main()
