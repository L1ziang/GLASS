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
# from data import CelebA
from resnet import resnet18
from model import Crop, DeNormalize, Face_Inversion, CELEBA_Dropout_Classifier, Face_Inversion_noise, CELEBA_disco_Classifier, Face_Disco_Inversion


import sys
sys.path.append('..')
from utils import *



def noisy_train(config, device, train_loader, optimizer_C, optimizer_P, inv_optimizer, epoch, dcor_weighting):
    config.model.train()
    config.inv_model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        target = target[:,2].view(-1,1).float()

        output, _ = config.model(config, data)

        optimizer_C.zero_grad()
        optimizer_P.zero_grad()
        inv_optimizer.zero_grad()

        config.model.adv_loss.backward()
        inv_optimizer.step()

        loss = F.binary_cross_entropy_with_logits(output, target)
        loss.backward()
        config.model.pruned_z_out.backward(config.model.pruned_z_in2.grad, retain_graph=True)
        config.model.unpruned_z_out.backward(config.model.unpruned_z_in.grad)

        optimizer_C.step()

        for params in config.model.noise_layer.parameters():
            params.grad *= (1-dcor_weighting)

        config.model.pruned_z_out.backward(-1 * dcor_weighting * config.model.pruned_z_in.grad)
        optimizer_P.step()

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
        config.model(config, data)
        feature = config.inv_model.hook.get_feature(config.inv_model.layer_id)

    inv_output = config.inv_model(feature)
    inv_loss = F.mse_loss(inv_output, data)
    inv_loss.backward()

    optimizer.step()

    if batch_idx % config.print_freq == 0:
        print("InvTrain Epoch: {} [{:.0f}%]\tLoss: {:.4f}".format(
              epoch, 100. * batch_idx / len_loader, inv_loss.item()))

    return batch_idx

def noisy_test(config, device, test_loader, n):
    config.model.eval()
    correct = 0
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            target = target[:,2].view(-1,1).float()

            output, _ = config.model(config, data)
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
    parser = argparse.ArgumentParser(description="Noisy Training Against Model Inversion Attack Example")
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
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': False}

    train_data = torch.load("./data/celeba_top3000ID_64_train.npz")
    test_data = torch.load("./data/celeba_top3000ID_64_test.npz")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, **cuda_kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=1024, shuffle=False, **cuda_kwargs)

    inv_data = torch.load("./data/celeba_other3000ID_64.npz")
    inv_train_loader = torch.utils.data.DataLoader(inv_data, batch_size=256, shuffle=True, **cuda_kwargs)

    print("len(adv_data):", len(train_data))
    print("len(inv_data):", len(inv_data))
    print("len(test_data):", len(test_data))



    model = None
    if config.arch == 'face_cnn':
        pretrained_model = resnet18(pretrained=False)
        pretrained_model.eval()
        pretrained_model.load_state_dict(torch.load('./models/celeba_3000_attractive.pth', map_location=device)['model'])
        print("pre-trained model loaded")
        model = CELEBA_disco_Classifier(pretrained_model, config.split_index, config.noise_scale)
    else:
        raise Exception("unknown model architecture")

    print("\ntarget model:", model, "\n")

    inv_model = None
    if config.inv:
        inv_model = Face_Disco_Inversion(config, model.cpu(), config.layers)
        print("\ninversion model:", inv_model, "\n")

        if config.load_model:
            model_path = os.path.join(config.model_dir, config.load_model)
            print('==> target model loading from {}'.format(model_path))
            model.load_state_dict(torch.load(model_path, map_location=device)['model'])

        # build attack dataset
        print('==> Building attack dataset')
        attack_test_data = Attack_Dataset()

        model.to(device)
        model.eval()
        
        for data, _ in test_loader:
            data = data.to(device)
            with torch.no_grad():
                model(config, data)
                feature = inv_model.hook.get_feat()
            feature = feature.to('cpu')
            data = data.to('cpu')
            attack_test_data.push(feature, data)

        attack_test_loader = torch.utils.data.DataLoader(attack_test_data, batch_size=128, shuffle=False, **cuda_kwargs)

        config.inv_model = inv_model.to(device)

    config.model = model.to(device)
    
    if config.dp:
        inv_model = Face_Inversion(pretrained_model.cpu(), config.layers)
        config.inv_model = inv_model.to(device)

        config.model = model.to(device)

    test_acc, test_loss = noisy_test(config, device, test_loader, 0)

    optimizer = None
    scheduler = None
    criterion = None
    if config.dp:
        if (config.optimizer == 'sgd'):
            exit()
            print("adv model using SGD optimizer")
            optimizer = optim.SGD(config.model.parameters(),config.lr,momentum=0.9,weight_decay=1e-4)
        elif (config.optimizer =='adam'):
            print("adv model using ADAM optimizer")
            optimizer_C = optim.Adam([{'params': model.front_layer.parameters()},
                        {'params': model.back_layer.parameters()},
                        {'params': model.stem.parameters()},
                        {'params': model.classifier.parameters()}], config.lr)
            optimizer_P = optim.Adam([{'params': model.noise_layer.parameters()}], config.lr)
            adv_optimizer = optim.Adam(config.inv_model.parameters(), config.lr)
            
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_C, T_max=config.epochs, eta_min=4e-08)

        config.inv_model.load_state_dict(torch.load('./models/celeba_inv_layer2_AdvUse.pth', map_location=device)['inv_model'])
        print("pre-trained inv_model loaded")

    inv_optimizer = None
    if config.inv:
        print("inv model using ADAM optimizer")
        inv_optimizer = optim.Adam(config.inv_model.parameters(), config.lr)

    
    start_epoch = 1
    save_path = os.path.join(config.model_dir, config.save_model)
    name = args.stage + args.index
    writer = SummaryWriter(os.path.join(config.log_dir, name))
    config.denorm = DeNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))


    for epoch in range(start_epoch, config.epochs+1):
        torch.manual_seed(666)

        if config.dp:
            noisy_train(config, device, train_loader, optimizer_C, optimizer_P, adv_optimizer, epoch, config.dcor_weighting)
            test_acc, test_loss = noisy_test(config, device, test_loader, n)

            writer.add_scalars('loss', {'test_loss': test_loss}, epoch)
            writer.add_scalars('accuracy', {'test_acc': test_acc}, epoch)

            state = {
                'epoch': epoch,
                'model': config.model.state_dict(),
                'best_acc': best_test_acc
            }
            torch.save(state, save_path)

        elif config.inv:
            len_inv_train = len(inv_train_loader)
            inv_train_enum = enumerate(inv_train_loader)
            for i in range(len_inv_train):
                inv_train(config, device, inv_train_enum, inv_optimizer, epoch, len_inv_train)
            inv_loss, ssim_v, psnr_v = inv_test(config, device, attack_test_loader, epoch, n)

            writer.add_scalars('loss', {'inv_loss': inv_loss}, epoch)
            writer.add_scalars('img quality', {'ssim_value': ssim_v, 'psnr_value': psnr_v}, epoch)

            if inv_loss < best_inv_loss:
                best_inv_loss = inv_loss
                state = {
                    'epoch': epoch,
                    'inv_model': config.inv_model.state_dict(),
                    'optimizer': inv_optimizer.state_dict(),
                    'best_inv_loss': best_inv_loss
                }
                print ("new best inv loss is {}, saving best inv model\n".format(best_inv_loss))
                torch.save(state, save_path)
                shutil.copyfile('recon_pics/{}/recon_{}.png'.format(n,epoch), 'recon_pics/{}/best_test.png'.format(n))

        if config.lr_scheduler == 'cosine':
            scheduler.step()
        elif config.lr_scheduler == 'sgd':
            if epoch == 20:
                config.lr/=10
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config.lr
        else:
            pass # it uses adam

    writer.close()
    print("=> Training Complete!\n")



if __name__ == '__main__':
    main()
