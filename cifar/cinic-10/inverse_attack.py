import argparse
import os
import shutil
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from torchvision import transforms
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from piq import ssim, psnr

from config import Config
from resnet import resnet18
from model import CIFAR10_Inversion, DeNormalize
import torchvision

import sys
sys.path.append('..')
from utils import *



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

            output = config.model(data)

            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

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
    inv_loss /= len(test_loader.dataset) * 3 * 32 * 32

    if config.adv:
        save_checkpoint(config, test_acc, inv_loss)

    print('\nTest set: test_acc: {:.2f}%, test_loss: {:.4f}, inv_loss: {:.4f}\n'
          .format(test_acc, test_loss, inv_loss))

    return test_acc, test_loss, inv_loss


def main():
    parser = argparse.ArgumentParser(description="Adversarial Training Against Model Inversion Attack Example")
    parser.add_argument('--config_file', type=str, default='', help="config file")
    parser.add_argument('--stage', type=str, default='', help="select the training stage")
    parser.add_argument('--index', type=str, default='', help="the index of the expriments")
    parser.add_argument('--seed', default=666, type=int)

    args = parser.parse_args()
    config = Config(args)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

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
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': True}

    cifar_mean = [0.4914, 0.4822, 0.4465]
    cifar_std = [0.2470, 0.2435, 0.2616]

    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar_mean,std=cifar_std)])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainset2 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    combined_dataset = torch.utils.data.ConcatDataset([trainset, trainset2])
    
    inv_train_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=1024, shuffle=True, **cuda_kwargs)
    cinic_directory = '../data/cinic'
    cinic_mean = [0.4749, 0.4691, 0.4255]
    cinic_std = [0.2406, 0.2366, 0.2574]

    cinic_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/valid',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
        batch_size=40, shuffle=True)

    print("len(test_data):", len(cinic_test))
    print("len(inv_data):", len(combined_dataset))

    model = None
    inv_model = None
    if config.arch == 'face_cnn':
        model = resnet18(pretrained=False)
    else:
        raise Exception("unknown model architecture")
    inv_model = CIFAR10_Inversion(model, config.layers)
    
    print("\ntarget model:", model, "\n")
    print("\ninversion model:", inv_model, "\n")

    ### for initialization experiments
    init = Init_Func(config.init_func)
    for name, W in model.named_parameters():
        if 'conv' in name and 'bias' not in name:
            W.data = init.init(W.data)

    config.model = model.to(device)
    config.inv_model = inv_model.to(device)

    # unlike resume, load model does not care optimizer status or start_epoch
    if config.load_model:
        model_path = os.path.join(config.model_dir, config.load_model)
        print('==> Loading from {}'.format(model_path))
        config.model.load_state_dict(torch.load(model_path, map_location=device)['model'])

    inv_optimizer = None
    scheduler = None

    if config.inv:
        print("inv model using ADAM optimizer")
        inv_optimizer = optim.Adam(config.inv_model.parameters(), config.lr)

    start_epoch = 1

    save_path = os.path.join(config.model_dir, config.save_model)
    name = args.stage + args.index
    writer = SummaryWriter(os.path.join(config.log_dir, name))

    config.denorm = DeNormalize(mean=cinic_mean,std=cinic_std)

    best_recon_loss = 999
    for epoch in range(start_epoch, config.epochs+1):

        len_inv_train = len(inv_train_loader)
        inv_train_enum = enumerate(inv_train_loader)

        if config.inv:
            for i in range(len_inv_train):
                inv_train(config, device, inv_train_enum, inv_optimizer, epoch, len_inv_train)

            _, _, inv_loss = test(config, device, cinic_test, epoch, n)
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
                shutil.copyfile('recon_pics/{}/recon_{}.png'.format(n,epoch), 'recon_pics/{}/best_test_inv.png'.format(n))

        if scheduler:
            scheduler.step()
        else:
            pass # it uses adam

    writer.close()
    print("=> Training Complete!\n")



if __name__ == '__main__':
    main()
