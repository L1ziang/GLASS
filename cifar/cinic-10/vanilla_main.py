import argparse
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from config import Config
from data import CIFAR10

from resnet import resnet18

def train(model, device, train_loader, optimizer, epoch, print_freq):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


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


def main():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Example")
    parser.add_argument('--config_file', type=str, default='', help="config file")
    parser.add_argument('--stage', type=str, default='', help="select the training stage")
    args = parser.parse_args()

    args = parser.parse_args()
    config = Config(args)

    torch.manual_seed(config.random_seed)

    os.makedirs(config.model_dir, exist_ok=True)

    use_cuda = torch.cuda.is_available() # use gpu as default
    device = torch.device(config.gpu if use_cuda else 'cpu')

    cuda_kwargs = {}
    if use_cuda:
        torch.cuda.set_device(config.gpu)
        cudnn.benchmark = True
        cuda_kwargs = {'num_workers': config.workers, 'pin_memory': True}

    cinic_directory = '../data/cinic'
    cinic_mean = [0.4749, 0.4691, 0.4255]
    cinic_std = [0.2406, 0.2366, 0.2574]
    train_dataset = torchvision.datasets.ImageFolder(cinic_directory + '/train',
    	transform=transforms.Compose([transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),  
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    test_dataset = torchvision.datasets.ImageFolder(cinic_directory + '/test',
    	transform=transforms.Compose([transforms.RandomCrop(32, padding=4),  
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)]))
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    cinic_train = torch.utils.data.DataLoader(
    combined_dataset,
    batch_size=1024, shuffle=True)

    cinic_test = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/valid',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=1000, shuffle=False)
    
    print("len(train_dataset): ", len(cinic_train))
    print("len(test_dataset): ", len(cinic_test))

    model = None
    if config.arch == 'cifar10_cnn':
        model = resnet18(pretrained=False)
    else:
        raise Exception("unknown model architecture")

    print("\nmodel:", model, "\n")

    config.model = model.to(device)

    optimizer = None
    if (config.optimizer == 'sgd'):
        print("using SGD optimizer")
        optimizer = optim.SGD(config.model.parameters(), config.lr,
                              momentum=0.9, weight_decay=1e-4)
    elif (config.optimizer == 'adam'):
        print("using ADAM optimizer")
        optimizer = optim.Adam(config.model.parameters(), config.lr)

    scheduler = None
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epochs,eta_min=4e-08)
    elif config.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=[int(config.epochs/2), int(config.epochs/4*3)])

    save_path = os.path.join(config.model_dir, config.save_model)
    writer = SummaryWriter(os.path.join(config.log_dir, args.stage))

    best_acc = 0.0
    for epoch in range(1, config.epochs+1):
        train(config.model, device, cinic_train, optimizer, epoch, config.print_freq)
        loss, acc = test(config.model, device, cinic_test)

        if scheduler:
            scheduler.step()
        else:
            pass # it uses adam

        writer.add_scalars('loss', {'test_loss': loss}, epoch)
        writer.add_scalars('accuracy', {'test_acc': acc}, epoch)

        if acc > best_acc:
            best_acc = acc
            state = {
                'epoch': epoch,
                'model': config.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_acc': best_acc,
            }
            torch.save(state, save_path)

    writer.close()
    print("=> Training Complete!\n")



if __name__ == '__main__':
    main()