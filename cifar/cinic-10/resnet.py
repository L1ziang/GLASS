import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch

import torch.nn.functional as F

__all__ = ['ResNet', 'resnet50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class fc_block(nn.Module):
    def __init__(self, inplanes, planes, drop_rate=0.2):
        super(fc_block, self).__init__()
        self.fc = nn.Linear(inplanes, planes)
        self.bn = nn.BatchNorm1d(planes)
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)
        self.relu = nn.ReLU(inplace=True)
        self.drop_rate = drop_rate

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        if self.drop_rate > 0:
            x = self.dropout(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):

    def __init__(self, block, layers, num_attributes=1, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)                                                                        #64,16,16
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                              #64, 16, 16 #64, 8, 8
        self.layer1 = self._make_layer(block, 64, layers[0])         # resnet50 50256*32*32 resnet18  64, 16, 16 #64, 8, 8
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)         #512*16*16            128, 8, 8  #128,4, 4
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)         #1024*8*8             256, 4, 4  #256,2, 2
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)         #2048*4*4             512, 2, 2  #512,1, 1
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                             #2048*1*1             512, 1, 1
        # self.stem = fc_block(512 * block.expansion, 512)                        #512
        self.classifier = nn.Linear(512*block.expansion, 10)
        # for i in range(num_attributes):
        #     setattr(self, 'classifier' + str(i).zfill(2), nn.Sequential(fc_block(512, 256), nn.Linear(256, 1)))
        self.num_attributes = num_attributes

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        y = self.classifier(x)

        return y
    
    def getLayerOutput(self, x, targetLayer):
        for name, module in self._modules.items():
            x = module(x)
            if targetLayer[0] == name:
                return x

class Encoder_network(nn.Module):

    def __init__(self, block, layers, attack_layer, num_attributes=1, zero_init_residual=False):
        super(Encoder_network, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)                              #64, 16, 16
        self.layer1 = self._make_layer(block, 64, layers[0])         # resnet50 50256*32*32 resnet18  64, 16, 16
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)         #512*16*16            128, 8, 8
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)         #1024*8*8             256, 4, 4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)         #2048*4*4             512, 2, 2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))                             #2048*1*1             512, 1, 1
        self.stem = fc_block(512 * block.expansion, 512)                        #512
        self.classifier = nn.Sequential(fc_block(512, 256), nn.Linear(256, 1))
        self.num_attributes = num_attributes
        self.attack_layer = attack_layer

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        if self.attack_layer[0] == 'layer3':
            x = self.layer4(x)

            x = self.avgpool(x)

            x = x.view(x.size(0), -1)
            x = self.stem(x)
            return x
        elif self.attack_layer[0] == 'layer4':
            x = self.avgpool(x)

            x = x.view(x.size(0), -1)
            x = self.stem(x)
            return x
    
    def getLayerOutput(self, x, targetLayer):
        for name, module in self._modules.items():
            x = module(x)
            if targetLayer[0] == name:
                return x


def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet50'])
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

def encoder_network(pretrained=False, progress=True, attack_layer=None, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
    #                **kwargs)
    model = Encoder_network(BasicBlock, [2, 2, 2, 2], attack_layer=attack_layer, **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet18'])
    return model

def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
    #                **kwargs)
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        init_pretrained_weights(model, model_urls['resnet34'])
    return model

def init_pretrained_weights(model, model_url):
    """
    Initialize model with pretrained weights.
    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    pretrain_dict = model_zoo.load_url(model_url)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)
    print("Initialized model with pretrained weights from {}".format(model_url))


if __name__ == '__main__':
    net = resnet18(pretrained=False)
    net.eval()
    x = torch.Tensor(1,3,32,32)
    for name, m in net.named_modules():
        print(name, m)
    y = net(x)
    print(y.shape)
    for name, module in net._modules.items():
        print(name)
    a = net.getLayerOutput(x, ['layer2'])
    print(a.shape)
        
    
    '''
    self.maxpool  torch.Size([1, 64, 16, 16])
    self.layer1  torch.Size([1, 64, 16, 16])
    self.layer2  torch.Size([1, 128, 8, 8])
    self.layer3  torch.Size([1, 256, 4, 4])
    self.layer4  torch.Size([1, 512, 2, 2])
    torch.Size([1, 1])
    
    conv1
    bn1
    relu
    maxpool
    layer1
    layer2
    layer3
    layer4
    avgpool
    stem
    classifier
    '''
    
    """ 
    celeba是128*128时
    self.layer1  torch.Size([1, 64, 32, 32])
    self.layer2  torch.Size([1, 128, 16, 16])
    self.layer3  torch.Size([1, 256, 8, 8])
    self.layer4  torch.Size([1, 512, 4, 4])
    self.avgpool  torch.Size([1, 512, 1, 1])
    """