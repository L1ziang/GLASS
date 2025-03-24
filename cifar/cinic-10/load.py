
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os
import torch.nn.functional as F
import torchvision

import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

import tqdm

class FaceScrub(Dataset):
    def __init__(self, root, transform=None, target_transform=None, train=True):
        #self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        #input = np.load(os.path.join(self.root, 'facescrub.npz'))
        input = np.load('./facescrub_150_150_3.npz')
        actor_images = input['images']
        actor_labels = input['labels']

        data = actor_images
        labels = actor_labels

        # v_min = data.min(axis=0)
        # v_max = data.max(axis=0)
        """np.savetxt("./filename1.txt",v_min)
        np.savetxt("./filename2.txt",v_max)
        exit()"""
        """print(data.shape) #(4170, 150, 150)
        print(v_min.shape) #(150, 150)
        print(v_min)
        print(v_max)"""
        """
        [[0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        ...
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]
        [0 0 0 ... 0 0 0]]
        [[255 255 255 ... 255 255 255]
        [255 255 255 ... 255 255 255]
        [255 255 255 ... 255 255 255]
        ...
        [255 255 255 ... 255 255 255]
        [255 255 255 ... 255 255 255]
        [255 255 255 ... 255 255 255]]"""

        np.random.seed(666) #将数据随机打乱
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        data = data[perm]
        labels = labels[perm]

        if train:
            self.data = data[0:int(0.9 * len(data))]
            self.labels = labels[0:int(0.9 * len(data))]
        else:
            self.data = data[int(0.9 * len(data)):]
            self.labels = labels[int(0.9 * len(data)):]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.uint8(img))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


def main():

    cinic_directory = '../data/cinic'

    train_dataset = torchvision.datasets.ImageFolder(cinic_directory + '/train',
    	transform=transforms.Compose([

        transforms.ToTensor()]))
    test_dataset = torchvision.datasets.ImageFolder(cinic_directory + '/test',
    	transform=transforms.Compose([
        transforms.ToTensor()]))
    combined_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])
    cinic_train = torch.utils.data.DataLoader(
    combined_dataset,
    batch_size=1024, shuffle=True)

    def get_mean_std(loader):
        # var[X] = E[X**2] - E[X]**2
        channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

        # for data, _ in tqdm(loader):
        for batch_idx, (data, target) in enumerate(loader):
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

        return mean, std

    mean, std = get_mean_std(cinic_train)
    print(mean)
    print(std)

    test_dataset = torchvision.datasets.ImageFolder(cinic_directory + '/valid',
    	transform=transforms.Compose([
        transforms.ToTensor()]))
    """train_set = transform(train_set)
    test_set = transform(test_set)"""
    torch.save(test_dataset, '../data/cinic-valid.npz')

if __name__ == '__main__':
    main()