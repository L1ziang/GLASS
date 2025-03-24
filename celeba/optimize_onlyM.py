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
from skip import *

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
    gen_model = skip(3, 3, num_channels_down = [32, 64, 128, 128, 128],
                                      num_channels_up =   [32, 64, 128, 128, 128],
                                      num_channels_skip = [4, 4, 4, 4, 4],
                                      filter_size_down = [7, 5, 5, 3, 3],  # type: ignore
                                      filter_size_up = [7, 5, 5, 3, 3],   # type: ignore
                      upsample_mode='nearest', downsample_mode='avg', need_sigmoid=True, pad='zero', act_fun='LeakyReLU') #.type(torch.cuda.FloatTensor)
   
    gen_model.cuda()
    return gen_model


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
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': False}

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
    
    inverseIter = iter(test_loader)
    for i in range(config.inverse_num):
        img_true, _ = next(inverseIter)
        img1 = config.denorm(img_true.detach()) 
        vutils.save_image(img1, 'recon_pics/{}/ref_{}_M.png'.format(n, i), normalize=False)
        img_true = img_true.to(device)
        
        with torch.no_grad():
            intermedia_feature = config.model.getLayerOutput(img_true, config.layers) #torch.Size([1, 128, 8, 8])
            
        gen_model = create_gen_model()
        rand_inp_og = torch.rand((1, 3, 64, 64)).detach().cuda()
        inp_noise = rand_inp_og.detach().clone()
        
        if config.optimizer =='adam':
            optimizer = optim.Adam(params = gen_model.parameters(), lr = config.lr, eps = 1e-3, amsgrad = config.AMSGrad)
        
        ys = gen_model(rand_inp_og)[:,:,:64, :64]
        for e in range(config.epochs):
            # add noise in input space
            if e < 1800:
                rand_inp = rand_inp_og + (inp_noise.normal_() * 10)
                for nn in [x for x in gen_model.parameters() if len(x) == 4]:
                    nn = nn + nn.detach().clone().normal_()*nn.std()/50
            elif e < 3300:
                rand_inp = rand_inp_og + (inp_noise.normal_() * 2)
                for nn in [x for x in gen_model.parameters() if len(x) == 4]:
                    nn = nn + nn.detach().clone().normal_()*nn.std()/50
            elif e < 6000:
                rand_inp = rand_inp_og + (inp_noise.normal_() / 2)
                for nn in [x for x in gen_model.parameters() if len(x) == 4]:
                    nn = nn + nn.detach().clone().normal_()*nn.std()/50
            else:
                rand_inp =  rand_inp_og
                
            # rand_inp = rand_inp_og
            rand_inp = rand_inp.cuda()
            ys = gen_model(rand_inp)[:,:,:64, :64]
            ys = norm(ys)
        
            optimizer.zero_grad()
            XFeature = config.model.getLayerOutput(ys, config.layers)
            featureLoss = ((intermedia_feature - XFeature)**2).mean()
            TVLoss = TV(ys)
            normLoss = l2loss(ys)
            totalLoss = featureLoss + config.lambda_TV * TVLoss + config.lambda_l2 * normLoss
            totalLoss.backward(retain_graph=True)
            optimizer.step()
            
        img2 = config.denorm(ys.cpu().detach()) # 保存原图
        vutils.save_image(img2, 'recon_pics/{}/recon_{}_M.png'.format(n, i), normalize=False)
        
            
    writer.close()
    print("=> Training Complete!\n")



if __name__ == '__main__':
    main()
