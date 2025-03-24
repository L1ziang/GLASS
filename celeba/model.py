from operator import inv
import torch
import torch.nn as nn
import torchvision.transforms.functional as F_1

import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

from resnet import resnet18

import numpy as np
import scipy.stats as st
import random

import torchvision
import torchvision.utils as vutils

class Crop(torch.nn.Module):

    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):

        return F_1.crop(img, self.top, self.left, self.height, self.width)


class Feature_Extractor():

    def __init__(self, model, layers, save_outputs=True):
        super().__init__()       
        self._layers = layers
        self.save_outputs = save_outputs
        self._features = {layer: torch.empty(0) for layer in layers}

        for layer_id in layers:
            layer = dict([*model.named_modules()])[layer_id]
            if self.save_outputs:
                layer.register_forward_hook(self._save_outputs_hook(layer_id))
            else:
                layer.register_forward_hook(self._save_inputs_hook(layer_id))
            print("For layer: {}, hook work".format(layer_id))

    def _save_inputs_hook(self, layer_id):
        def fn(_, input, output):
            self._features[layer_id] = input[0]
        return fn

    def _save_outputs_hook(self, layer_id):
        def fn(_, input, output):
            self._features[layer_id] = output
        return fn

    def get_features(self):
        return self._features

    def get_feature(self, layer_id):
        return self._features[layer_id]

    def get_feat(self):
        return self._features[self._layers[0]]



class DeNormalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
            tensor (Tensor): Normalized Tensor image.

        Returns:
            Tensor: Denormalized Tensor.
        """
        return self._denormalize(tensor)

    def _denormalize(self, tensor):
        tensor = tensor.clone()

        dtype = tensor.dtype
        mean = torch.as_tensor(self.mean, dtype=dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=dtype, device=tensor.device)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        # tensor.sub_(mean).div_(std)
        tensor.mul_(std).add_(mean)

        return tensor


class Face_Inversion(nn.Module):
    def __init__(self, model, layers, aux_model=None):
        super(Face_Inversion, self).__init__()
        self.layer_id = layers[0]
        self.image_shape = torch.Size([3,64,64])
        self.hook = Feature_Extractor(model, layers)
        if aux_model:
            self.aux_hook = Feature_Extractor(aux_model, layers)
        self.decoder = self._create_decoder(model, layers[0])
        self.dense = self._create_dense(model, layers[0])

    def forward(self, x):
        B = x.shape[0]
        
        if self.layer_id == "stem":
            x = self.dense(x)
            x = x.view(B, -1, 1, 1)
        x = self.decoder(x)

        return x

    def _create_dense(self, model, layer_id):
        layers = []
        layers += [nn.Sequential(nn.Linear(512, 2048), nn.ReLU())]
        return nn.Sequential(*layers)


    def _create_decoder(self, model, layer_id):

        def make_layers(in_channels, config):
            layers = []

            for out_channels in config:
                upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
                layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
                in_channels = out_channels
            upconv2d = nn.ConvTranspose2d(in_channels, 3, 4, 2, 1)
            layers += [upconv2d, nn.Tanh()]

            return nn.Sequential(*layers)
        
        def make_layers_avgpool(in_channels, config):
            layers = []

            layers += [nn.Upsample(scale_factor=2)]
            for out_channels in config:
                upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
                layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
                in_channels = out_channels
            upconv2d = nn.ConvTranspose2d(in_channels, 3, 4, 2, 1)
            layers += [upconv2d, nn.Tanh()]

            return nn.Sequential(*layers)

        with torch.no_grad():
            model.eval()
            x = torch.randn(1, 3, 64, 64)#.cuda() # it depends
            model(x)
            feature = self.hook.get_feature(layer_id)
            
        self.feature_shape = feature.shape
        channels = feature.shape[1]
        if self.layer_id == "stem":
            decoder_config =[1024, 512, 256, 128, 64]
            return make_layers_avgpool(2048, decoder_config)
        height = feature.shape[2]
        width = feature.shape[3]

        decoder_config = None
        if height*width == 32*32:
            # decoder_config = [64]
            decoder_config = []
        elif height*width == 16*16: # changed for 64
            decoder_config = [64]
        elif height*width == 8*8: # changed for 64
            decoder_config = [128, 64]
        elif height*width == 4*4: # changed for 64
            decoder_config = [256, 128, 64]
            # decoder_config = [1024, 512, 256, 64]
        elif height*width == 2*2: # changed for 64
            decoder_config = [512, 256, 128, 64]
            # decoder_config = [1024, 512, 256, 64]
        elif height*width == 1*1: # changed for 64
            decoder_config =[512, 256, 128, 64]
            return make_layers_avgpool(channels, decoder_config)
        else:
            raise Exception("The shape of target_feature is not supported")

        return make_layers(channels, decoder_config)
    
class Face_Disco_Inversion(nn.Module):
    def __init__(self, config, model, layers, aux_model=None):
        super(Face_Disco_Inversion, self).__init__()
        self.layer_id = layers[0]
        self.image_shape = torch.Size([3,64,64])
        self.hook = Feature_Extractor(model, layers)
        if aux_model:
            self.aux_hook = Feature_Extractor(aux_model, layers)
        self.decoder = self._create_decoder(config, model, layers[0])
        self.dense = self._create_dense(model, layers[0])

    def forward(self, x):
        B = x.shape[0]
        
        if self.layer_id == "stem":
            x = self.dense(x)
            x = x.view(B, -1, 1, 1)
        x = self.decoder(x)

        return x

    def _create_dense(self, model, layer_id):
        layers = []
        layers += [nn.Sequential(nn.Linear(512, 2048), nn.ReLU())]
        return nn.Sequential(*layers)


    def _create_decoder(self, config, model, layer_id):

        def make_layers(in_channels, config):
            layers = []

            for out_channels in config:
                upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
                layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
                # layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.ReLU(True)]
                in_channels = out_channels
            upconv2d = nn.ConvTranspose2d(in_channels, 3, 4, 2, 1)
            layers += [upconv2d, nn.Tanh()]

            return nn.Sequential(*layers)
        
        def make_layers_avgpool(in_channels, config):
            layers = []

            layers += [nn.Upsample(scale_factor=2)]
            for out_channels in config:
                upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
                layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
                in_channels = out_channels
            upconv2d = nn.ConvTranspose2d(in_channels, 3, 4, 2, 1)
            layers += [upconv2d, nn.Tanh()]

            return nn.Sequential(*layers)

        with torch.no_grad():
            model.eval()
            x = torch.randn(1, 3, 64, 64)#.cuda() # it depends
            model(config, x)
            feature = self.hook.get_feature(layer_id)
            
        self.feature_shape = feature.shape
        channels = feature.shape[1]
        if self.layer_id == "stem":
            decoder_config =[1024, 512, 256, 128, 64]
            return make_layers_avgpool(2048, decoder_config)
        height = feature.shape[2]
        width = feature.shape[3]

        decoder_config = None
        if height*width == 32*32:
            exit()
            # decoder_config = [64]
        elif height*width == 16*16: # changed for 64
            decoder_config = [64]
        elif height*width == 8*8: # changed for 64
            decoder_config = [128, 64]
        elif height*width == 4*4: # changed for 64
            decoder_config = [256, 128, 64]
            # decoder_config = [1024, 512, 256, 64]
        elif height*width == 2*2: # changed for 64
            decoder_config = [512, 256, 128, 64]
            # decoder_config = [1024, 512, 256, 64]
        elif height*width == 1*1: # changed for 64
            decoder_config =[512, 256, 128, 64]
            return make_layers_avgpool(channels, decoder_config)
        else:
            raise Exception("The shape of target_feature is not supported")

        return make_layers(channels, decoder_config)

class Face_Inversion_noise(nn.Module):
    def __init__(self, model, layers, aux_model=None):
        super(Face_Inversion_noise, self).__init__()
        self.layer_id = layers[0]
        self.image_shape = torch.Size([3,64,64])
        self.hook = Feature_Extractor(model, layers)
        if aux_model:
            self.aux_hook = Feature_Extractor(aux_model, layers)
        self.decoder = self._create_decoder(model, layers[0])
        self.dense = self._create_dense(model, layers[0])

    def forward(self, x):
        B = x.shape[0]
        
        if self.layer_id == "stem":
            x = self.dense(x)
            x = x.view(B, -1, 1, 1)
        x = self.decoder(x)

        return x
        
    
    def _create_dense(self, model, layer_id):
        layers = []
        layers += [nn.Sequential(nn.Linear(512, 2048), nn.ReLU())]
        return nn.Sequential(*layers)


    def _create_decoder(self, model, layer_id):

        def make_layers(in_channels, config):
            layers = []

            for out_channels in config:
                upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
                layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
                # layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.ReLU(True)]
                in_channels = out_channels
            upconv2d = nn.ConvTranspose2d(in_channels, 3, 4, 2, 1)
            layers += [upconv2d, nn.Tanh()]

            return nn.Sequential(*layers)
        
        def make_layers_avgpool(in_channels, config):
            layers = []

            layers += [nn.Upsample(scale_factor=2)]
            for out_channels in config:
                upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
                layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.2, inplace=True)]
                # layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.ReLU(True)]
                in_channels = out_channels
            upconv2d = nn.ConvTranspose2d(in_channels, 3, 4, 2, 1)
            layers += [upconv2d, nn.Tanh()]

            return nn.Sequential(*layers)

        with torch.no_grad():
            model.eval()
            x = torch.randn(1, 3, 64, 64)#.cuda() # it depends
            model(x)
            feature = self.hook.get_feature(layer_id)
            
        self.feature_shape = feature.shape
        channels = feature.shape[1]
        if self.layer_id == "stem":
            decoder_config =[1024, 512, 256, 128, 64]
            return make_layers_avgpool(2048, decoder_config)
        height = feature.shape[2]
        width = feature.shape[3]

        decoder_config = None
        if height*width == 32*32:
            exit()
            # decoder_config = [64]
        elif height*width == 16*16: # changed for 64
            decoder_config = [64]
        elif height*width == 8*8: # changed for 64
            decoder_config = [128, 64]
        elif height*width == 4*4: # changed for 64
            decoder_config = [256, 128, 64]
            # decoder_config = [1024, 512, 256, 64]
        elif height*width == 2*2: # changed for 64
            decoder_config = [512, 256, 128, 64]
            # decoder_config = [1024, 512, 256, 64]
        elif height*width == 1*1: # changed for 64
            decoder_config =[512, 256, 128, 64]
            return make_layers_avgpool(channels, decoder_config)
        else:
            raise Exception("The shape of target_feature is not supported")

        return make_layers(channels, decoder_config)



class Face_Transformer(nn.Module):

    def __init__(self, in_channels, layers):
        super( Face_Transformer, self).__init__()

        conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        if layers[0] == 'stem':
            self.encoder = nn.Sequential(nn.Linear(512, 512), nn.ReLU())
        else:
            self.encoder = nn.Sequential(
                conv2d,
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                conv2d,
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
                conv2d,
                nn.BatchNorm2d(in_channels),
                nn.ReLU(True),
            )

    def forward(self, x):
        
        x = self.encoder(x)

        return x
    
class Crop(torch.nn.Module):

    def __init__(self, top, left, height, width):
        super().__init__()
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def forward(self, img):

        return F_1.crop(img, self.top, self.left, self.height, self.width)
    
class Laplace_Noise_Activation(nn.Module):

    def __init__(self, noise_scale=0.1):
        super(Laplace_Noise_Activation, self).__init__()
        if noise_scale > 0.0:
            self.set_noise(noise_scale)
        else:
            raise ValueError("noise scale must be positive number")

    def set_noise(self, scale: float):
        # self._noise = torch.distributions.Laplace(0.0, scale)
        self._noise = torch.distributions.Laplace(torch.tensor(0.0).to(device='cuda'), torch.tensor(scale).to(device='cuda'))

    def forward(self, input: torch.Tensor):
        # return input + self._noise.sample(input.size())
        return input + self._noise.sample(input.size()).to(input.device)


class CELEBA_DP_Classifier(nn.Module):

    def __init__(self, model, split_index, noise_scale):
        super(CELEBA_DP_Classifier, self).__init__()
        self._split_index = split_index

        if split_index == 2:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            back_layer = [model.layer3, model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Laplace_Noise_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 3:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3]
            back_layer = [model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Laplace_Noise_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 4:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4]
            back_layer = [model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Laplace_Noise_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 5:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Laplace_Noise_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 6:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Laplace_Noise_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier

    def forward(self, x):
        
        if self._split_index == 6:
            inter_feat = self.front_layer(x)
            # out = self.noise_layer(inter_feat)
            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.noise_layer(out)
            out = self.classifier(out)

            return out, inter_feat
        else:
            inter_feat = self.front_layer(x)
            out = self.noise_layer(inter_feat)
            out = self.back_layer(out)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.classifier(out)

            return out, inter_feat
        
class Dropout_Activation(nn.Module):
    #for Dropout, the noise_scale is the dropout rate
    def __init__(self, noise_scale=0.5):
        super(Dropout_Activation, self).__init__()
        self.noise_scale=noise_scale
        if noise_scale < 0.0 or noise_scale > 1.0:
            raise ValueError("Dropout noise scale must be between 0 and 1")

    def forward(self, input: torch.Tensor):
        if self.noise_scale == 1:
            return torch.zeros_like(input)
        elif self.noise_scale == 0:
            return input
        else:
            noise_mask = ( torch.rand(input.shape) > self.noise_scale).float()
            return input * noise_mask.to(input.device)


class CELEBA_Dropout_Classifier(nn.Module):

    def __init__(self, model, split_index, noise_scale):
        super(CELEBA_Dropout_Classifier, self).__init__()
        self._split_index = split_index

        if split_index == 2:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            back_layer = [model.layer3, model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 3:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3]
            back_layer = [model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 4:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4]
            back_layer = [model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 5:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 6:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier

    def forward(self, x):
        
        if self._split_index == 6:
            inter_feat = self.front_layer(x)
            # out = self.noise_layer(inter_feat)
            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.noise_layer(out)
            out = self.classifier(out)

            return out, inter_feat
        else:
            inter_feat = self.front_layer(x)
            out = self.noise_layer(inter_feat)
            out = self.back_layer(out)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.classifier(out)

            return out, inter_feat
        
class PruningNetwork(nn.Module):
    """ Nothing special about the pruning model,
    it is a standard resnet predictive model. Might update it later
    """
    def __init__(self, pruning_ratio=0.5):
        super(PruningNetwork, self).__init__()
        self.pruning_ratio = pruning_ratio #config["pruning_ratio"]
        self.pruning_style = "learnable" #config["pruning_style"]

        if self.pruning_style == "learnable":
            self.temp = 1/30
            self.logits = 128 #config["channels"] for layer2 : 128
            self.split_layer = 6 #config["split_layer"] for layer2 : split_layer=6
            self.model = torchvision.models.resnet18(pretrained=False)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Flatten(),
                                          nn.Linear(num_ftrs, self.logits))

            self.model = nn.ModuleList(list(self.model.children())[self.split_layer:])
            self.model = nn.Sequential(*self.model)
        elif self.pruning_style == "random":
            # decoy layer to allow creation of optimizer
            self.decoy_layer = nn.Linear(10, 10)

    def prune_channels(self, z, indices=None):
        # Indexing is an inplace operation which creates problem during backprop hence z is cloned first
        z = z.clone()
        z[:, indices] = 0.
        return z

    @staticmethod
    def get_random_channels(x, ratio):
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        channels_to_prune = torch.randperm(x.shape[1], device=x.device)[:num_prunable_channels]
        return channels_to_prune

    def custom_sigmoid(self, x, offset):
        exponent = (x - offset) / self.temp
        #answer = (1 / (1 + torch.exp( - exponent / self.temp)))
        answer = nn.Sigmoid()(exponent)
        return answer

    def get_channels_from_network(self, x, ratio):
        fmap_score = self.network_forward(x)
        num_channels = x.shape[1]
        num_prunable_channels = int(num_channels * ratio)
        threshold_score = torch.sort(fmap_score)[0][:, num_prunable_channels].unsqueeze(1)
        fmap_score = self.custom_sigmoid(fmap_score, threshold_score)
        return fmap_score

    def network_forward(self, x):
        return self.model(x)

    def forward(self, x):
        if self.pruning_style == "random":
            exit()
            indices = self.get_random_channels(x, self.pruning_ratio)
            x = self.prune_channels(x, indices)
        elif self.pruning_style == "learnable":
            # get score for feature maps
            channel_score = self.get_channels_from_network(x, self.pruning_ratio)
            x = x*channel_score.unsqueeze(-1).unsqueeze(-1)
        return x
        
class CELEBA_disco_Classifier(nn.Module):

    def __init__(self, model, split_index, noise_scale):
        super(CELEBA_disco_Classifier, self).__init__()
        self._split_index = split_index

        if split_index == 2:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            back_layer = [model.layer3, model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = PruningNetwork(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 3:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3]
            back_layer = [model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = PruningNetwork(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 4:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4]
            back_layer = [model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = PruningNetwork(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 5:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = PruningNetwork(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 6:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = PruningNetwork(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier

    def forward(self, config, x):

        if config.inv:
            out = self.front_layer(x)
            inter_feat = self.noise_layer(out)

            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.classifier(out)

            return out, inter_feat
        
        if self._split_index == 6:
            pass
            exit()
            inter_feat = self.front_layer(x)
            # out = self.noise_layer(inter_feat)
            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.noise_layer(out)
            out = self.classifier(out)

            return out, inter_feat
        else:
            if config.dp:
                self.unpruned_z_out = self.front_layer(x)
                self.unpruned_z_in = self.unpruned_z_out.detach()
                self.unpruned_z_in.requires_grad = True
                self.pruned_z_out = self.noise_layer(self.unpruned_z_in)
                # self.pruned_z_out = self.unpruned_z_in

                self.pruned_z_in = self.pruned_z_out.detach()
                self.pruned_z_in.requires_grad = True

                # if config.dp:
                x_recons = config.inv_model(self.pruned_z_in)
                self.adv_loss = F.mse_loss(x_recons, x)

                self.pruned_z_in2 = self.pruned_z_out.detach()
                self.pruned_z_in2.requires_grad = True

                # inter_feat = self.front_layer(x)
                # out = self.noise_layer(self.pruned_z_out)
                out = self.back_layer(self.pruned_z_in2)
                out = torch.flatten(out, 1)
                out = self.stem(out)
                out = self.classifier(out)

                return out, self.pruned_z_in2
            else:
                inter_feat = self.front_layer(x)
                out = self.noise_layer(inter_feat)
                out = self.back_layer(out)
                out = torch.flatten(out, 1)
                out = self.stem(out)
                # out = self.noise_layer(out)
                out = self.classifier(out)

                return out, inter_feat
            
class CELEBA_disco_Classifier_inv(nn.Module):

    def __init__(self, model, split_index, noise_scale):
        super(CELEBA_disco_Classifier_inv, self).__init__()
        self._split_index = split_index

        if split_index == 2:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            back_layer = [model.layer3, model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = PruningNetwork(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier

    def forward(self, x):

            out = self.front_layer(x)
            inter_feat = self.noise_layer(out)

            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.classifier(out)

            return out, inter_feat

        
class CELEBA_NoPeek_Classifier(nn.Module):

    def __init__(self, model, split_index):
        super(CELEBA_NoPeek_Classifier, self).__init__()
        self._split_index = split_index

        if split_index == 2:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            back_layer = [model.layer3, model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            # self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 3:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3]
            back_layer = [model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            # self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 4:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4]
            back_layer = [model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            # self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 5:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            # self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 6:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            # self.noise_layer = Dropout_Activation(noise_scale)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier

    def forward(self, x):
        
        if self._split_index == 6:
            inter_feat = self.front_layer(x)
            # out = self.noise_layer(inter_feat)
            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            # out = self.noise_layer(out)
            out = self.classifier(out)

            return out, inter_feat
        else:
            inter_feat = self.front_layer(x)
            # out = self.noise_layer(inter_feat)
            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.classifier(out)

            return out, inter_feat

class ShredderNoisyActivation(nn.Module):
    def __init__(self, activation_size, dist_of_noise="norm",
                 loc=0.0, scale=20.0):
        super(ShredderNoisyActivation, self).__init__()

        self.activation_size = activation_size
        self.loc = loc
        self.scale = scale
        if dist_of_noise == "norm":
            self.dist_of_noise = st.norm
        elif dist_of_noise == "laplace":
            self.dist_of_noise = st.laplace

        # initialize the noise tensor
        self.train_noise = None
        self._initialize_noise_tensor()

        self.dist_params_of_noise = None
        self.sorted_noise_index = None
        self.eval_noise = []
        self.sampled_noises_ds = []

    def forward(self, input):
        if self.training:
            return self._forward_train(input)
        else:
            return self._forward_eval(input)

    def _forward_train(self, input):
        return input + self.train_noise

    def _forward_eval(self, input):
        self.sample_noise_tensor(input.size(0))
        A = torch.stack(self.eval_noise)
        if input.device.type == 'cuda':

            return input + A
        else:
            return input + A.cpu()

    def _initialize_noise_tensor(self):
        m = torch.distributions.laplace.Laplace(
            loc=self.loc, scale=self.scale, validate_args=None)
        self.train_noise = nn.Parameter(m.rsample(self.activation_size))

    def sample_noise_tensor(self, batch):
        self.eval_noise = []
        for _ in range(int(batch)):
            # flatten the noise optimized during training
            np_train_noise = self.train_noise.clone().detach().cpu().numpy()
            noise_flatten = np_train_noise.flatten()
            # sample new noise from fitted distriution
            index = random.randint(0, len(self.sampled_noises_ds)-1)
            # index = 1
            noise_sampled = self.dist_of_noise.rvs(
                loc=self.sampled_noises_ds[index][0][-2],
                scale=self.sampled_noises_ds[index][0][-1],
                size=np.prod(self.activation_size))
            # reorder and reshape the new noise
            sorted_sampled_noise_index = np.argsort(noise_sampled)
            noise_flatten[self.sampled_noises_ds[index][1]] =\
                noise_sampled[sorted_sampled_noise_index]
            updated_noise = noise_flatten.reshape(self.activation_size)

            self.eval_noise.append(nn.Parameter(torch.Tensor(updated_noise)).cuda())
    
    def sample_noise_distribution(self):

        # flatten the noise optimized during training
        np_train_noise = self.train_noise.clone().detach().cpu().numpy()
        noise_flatten = np_train_noise.flatten()

        # get the order of noise elements
        self.sorted_noise_index = np.argsort(noise_flatten)

        # fit the noise to distriution
        self.dist_params_of_noise = self.dist_of_noise.fit(noise_flatten)

        self.sampled_noises_ds.append([self.dist_params_of_noise, self.sorted_noise_index])
        

class CELEBA_cloak_Classifier(nn.Module):

    def __init__(self, model, split_index):
        super(CELEBA_cloak_Classifier, self).__init__()
        self._split_index = split_index

        loc=0.0
        scale=20.0
        
        dist_of_noise="laplace"

        self.loc = loc
        self.scale = scale
        self.intermidiate_shape = (128, 8, 8) # layer2
        self.dist_of_noise = dist_of_noise


        if split_index == 2:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2]
            back_layer = [model.layer3, model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = ShredderNoisyActivation(self.intermidiate_shape, loc=self.loc, scale=self.scale, dist_of_noise=dist_of_noise)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 3:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3]
            back_layer = [model.layer4, model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = ShredderNoisyActivation(self.intermidiate_shape, loc=self.loc, scale=self.scale, dist_of_noise=dist_of_noise)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 4:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4]
            back_layer = [model.avgpool]
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = ShredderNoisyActivation(self.intermidiate_shape, loc=self.loc, scale=self.scale, dist_of_noise=dist_of_noise)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 5:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = ShredderNoisyActivation(self.intermidiate_shape, loc=self.loc, scale=self.scale, dist_of_noise=dist_of_noise)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier
        elif split_index == 6:
            front_layer = [model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3, model.layer4, model.avgpool]
            back_layer = []
            self.front_layer = nn.Sequential(*front_layer)
            self.noise_layer = ShredderNoisyActivation(self.intermidiate_shape, loc=self.loc, scale=self.scale, dist_of_noise=dist_of_noise)
            self.back_layer = nn.Sequential(*back_layer)
            self.stem = model.stem
            self.classifier = model.classifier

    # def forward(self, x, origin, n, i, i1):
    def forward(self, x):
        
        if self._split_index == 6:
            pass
            exit()
            inter_feat = self.front_layer(x)
            # out = self.noise_layer(inter_feat)
            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.noise_layer(out)
            out = self.classifier(out)

            return out, inter_feat
        else:
            out = self.front_layer(x)
            out.requeires_grad = True
            # inter_feat = self.noise_layer(out, origin, n, i, i1)
            inter_feat = self.noise_layer(out) 
            out = self.back_layer(inter_feat)
            out = torch.flatten(out, 1)
            out = self.stem(out)
            out = self.classifier(out)

            return out, inter_feat

if __name__ == '__main__':

    model = CELEBA_disco_Classifier(resnet18(pretrained=False), 2, 0.5)
    model = nn.ModuleList(list(model.children())[1:2])
    model = nn.Sequential(*model)
    print(list(model.children()))
    exit()
    x = torch.Tensor(1, 128, 8, 8)
    M = torchvision.models.resnet18(pretrained=False,)
    num_ftrs = M.fc.in_features
    M.fc = nn.Sequential(nn.Flatten(),
                                    nn.Linear(num_ftrs, 128))
    model = nn.ModuleList(list(M.children())[6:])
    model = nn.Sequential(*model)
    print(list(model.children()))
    y = model(x)
    print(y.shape)
                   
    exit()
    model = CELEBA_DP_Classifier(resnet18(pretrained=False), 2, 0.5)
    for i in model.named_modules():
        print(i)
    exit()
    inv_model = Face_Inversion(model, ['avgpool'])
    x = torch.Tensor(1,512,1,1)
    y = inv_model(x)
    print(y.shape)