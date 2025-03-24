import torch
import torch.nn as nn
import torch.nn.functional as F


r"""
    using vgg-16 with batchnorm
"""
vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class CIFAR10_Classifier(nn.Module):

    def __init__(self):
        super(CIFAR10_Classifier, self).__init__()
        self.features = self._create_features(vgg16_cfg, batch_norm=True)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _create_features(self, cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if batch_norm:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False)
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(True)]
                else:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                    layers += [conv2d, nn.ReLU(True)]
                in_channels = v

        return nn.Sequential(*layers)


class Laplace_Noise_Activation(nn.Module):

    def __init__(self, noise_scale=0.1):
        super(Laplace_Noise_Activation, self).__init__()
        if noise_scale > 0.0:
            self.set_noise(noise_scale)
        else:
            raise ValueError("noise scale must be positive number")

    def set_noise(self, scale: float):
        self._noise = torch.distributions.Laplace(0.0, scale)

    def forward(self, input: torch.Tensor):
        return input + self._noise.sample(input.size()).to(input.device)


class CIFAR10_DP_Classifier(nn.Module):

    def __init__(self, model, split_index, noise_scale):
        super(CIFAR10_DP_Classifier, self).__init__()

        front_layer = list(model.features)[:split_index]
        back_layer = list(model.features)[split_index:]

        self.front_layer = nn.Sequential(*front_layer)
        self.noise_layer = Laplace_Noise_Activation(noise_scale)
        self.back_layer = nn.Sequential(*back_layer)
        self.classifier = model.classifier

    def forward(self, x):
        inter_feat = self.front_layer(x)
        out = self.noise_layer(inter_feat)
        out = self.back_layer(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out, inter_feat


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


class CIFAR10_Inversion(nn.Module):
    def __init__(self, model, layers, aux_model=None, aux_layers=None, out_hook=True):
        super(CIFAR10_Inversion, self).__init__()

        self.image_shape = torch.Size([3,32,32])
        self.layer_id = layers[0]
        self.hook = Feature_Extractor(model, layers, out_hook)
        self.decoder = self._create_decoder(model)

    def forward(self, x):

        x = self.decoder(x)

        return x

    def _create_decoder(self, model):

        def make_layers(in_channels, config):
            layers = []

            for out_channels in config:
                upconv2d = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False)
                layers += [upconv2d, nn.BatchNorm2d(out_channels), nn.LeakyReLU(0.3, inplace=True)]
                in_channels = out_channels
            upconv2d = nn.ConvTranspose2d(in_channels, 3, 3, 1, 1)
            layers += [upconv2d, nn.Tanh()]

            return nn.Sequential(*layers)

        model.eval()
        with torch.no_grad():
            x = torch.randn(1, 3, 32, 32)
            model(x)
            feature = self.hook.get_feat()
        self.feature_shape = feature.shape
        channels = feature.shape[1]
        height = feature.shape[2]
        width = feature.shape[3]

        decoder_config = None
        if height*width == 32*32:
            decoder_config = []
        elif height*width == 16*16:
            decoder_config = [64]
        elif height*width == 8*8:
            decoder_config = [128, 64]
        elif height*width == 4*4:
            decoder_config = [256, 128, 64]
        elif height*width == 2*2:
            decoder_config = [512, 256, 128, 64]
        elif height*width == 1*1:
            decoder_config = [1024, 512, 256, 128, 64]
        else:
            raise Exception("The shape of target_feature is not supported")

        return make_layers(channels, decoder_config)


class CIFAR10_Transformer(nn.Module):

    def __init__(self, in_channels):
        super(CIFAR10_Transformer, self).__init__()

        self.in_channels = in_channels

        conv2d = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)

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
        # x = self.decoder(x)

        return x

r"""
    DeepSiM GAN from here
"""

cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
       512, 512, 512, 'M']

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AdaptiveAvgPool2d((1, 1))]
        return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.3, inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
            )

    def forward(self, x):
        return self.upsample(x)


class Comparator(nn.Module):
    def __init__(self, model, target_layer):
        super(Comparator, self).__init__()
        features = list(model.features)[:target_layer]
        self.features = nn.ModuleList(features)
        self.target_layer = target_layer

    def forward(self, x):
        if self.target_layer == -1:
            return x
        else:
            for model in self.features:
                x = model(x)
            return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.3, inplace=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.classifier(x)
        return x



if __name__ == '__main__':

    r"""
    Test inversion model
    """
    net = CIFAR10_Classifier()
    for name, m in net.named_modules():
        print(name, m)

    r"""
    Test GAN
    """
