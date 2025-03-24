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
# from Experiment_celeba_Attack.celeba.data_celeba_1000 import CelebA
from resnet import resnet18
from model import Crop, Face_Inversion, DeNormalize
from piq import ssim, psnr, LPIPS

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

def main():
    parser = argparse.ArgumentParser(description="optimize_onlyX")
    parser.add_argument('--config_file', type=str, default='', help="config file")
    parser.add_argument('--stage', type=str, default='', help="select the training stage")
    parser.add_argument('--index', type=str, default='', help="the index of the expriments")
    args = parser.parse_args()

    config = Config(args)

    torch.manual_seed(config.random_seed)

    os.makedirs(config.model_dir, exist_ok=True)
    n = config.layers[0] + '_' + args.index
    os.makedirs('recon_pics/{}'.format(n), exist_ok=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device(config.gpu if use_cuda else 'cpu')

    cuda_kwargs = {}
    if use_cuda:
        torch.cuda.set_device(config.gpu)
        cudnn.benchmark = True
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': True}

    celeba_transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    Crop(70,35,128,128),
                                    transforms.Resize([64, 64]),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

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
    loss, acc = test(config.model, device, test_loader)
    print("acc : ", acc)
    
    
    inverseIter = iter(test_loader)
    for i in range(config.inverse_num):
        img_true, _ = next(inverseIter)
        img1 = config.denorm(img_true.detach())
        vutils.save_image(img1, 'recon_pics/{}/ref_{}.png'.format(n, i), normalize=False)
        img_true = img_true.to(device)
        
        with torch.no_grad():
            intermedia_feature = config.model.getLayerOutput(img_true, config.layers) #torch.Size([1, 128, 8, 8])
        X = torch.randn(img_true.size(), requires_grad = True, device="cuda")
        if config.optimizer =='adam':
            optimizer = optim.Adam(params = [X], lr = config.lr, eps = 1e-3, amsgrad = config.AMSGrad)
        
        for e in range(config.epochs):
            XFeature = config.model.getLayerOutput(X, config.layers)
            featureLoss = ((intermedia_feature - XFeature)**2).mean()
            TVLoss = TV(X)
            normLoss = l2loss(X)
            totalLoss = featureLoss + config.lambda_TV * TVLoss #+ config.lambda_l2 * normLoss
            optimizer.zero_grad()
            totalLoss.backward(retain_graph=True)
            optimizer.step()
            print ("Iter ", e, "Feature loss: ", featureLoss.cpu().detach().numpy(), "TVLoss: ", TVLoss.cpu().detach().numpy(), "l2Loss: ", normLoss.cpu().detach().numpy())
            writer.add_scalars('1', {'Feature loss': featureLoss}, e)
            writer.add_scalars('2', {'TV Loss': TVLoss}, e)
            writer.add_scalars('3', {'l2 Loss': normLoss}, e)
            
        img2 = config.denorm(X.cpu().detach()) # 保存原图
        vutils.save_image(img2, 'recon_pics/{}/recon_{}.png'.format(n, i), normalize=False)

    writer.close()
    print("=> Training Complete!\n")



if __name__ == '__main__':
    main()
