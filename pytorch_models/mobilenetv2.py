''' MobileNetV2 in PyTorch. '''
''' Oficial paper at https://arxiv.org/abs/1801.04381 '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Linear_act,return_activation,apply_conv,apply_linear,apply_pool, MamasitaNetwork,add_gaussian,apply_DePool,apply_DeConv

            # (expansion, out_planes, num_blocks, stride)
cfg = {
    "MobileNetStandard": [(1,  16, 1, 1),
                          (6,  24, 2, 1),
                          (6,  32, 3, 2),
                          (6,  64, 4, 2),
                          (6,  96, 3, 1),
                          (6, 160, 3, 2),
                          (6, 320, 1, 1)],
    "MobileNetSmallv0": [(1,  16, 1, 1),
                         (6,  24, 2, 2),
                         (6,  32, 2, 2),
                         (6,  64, 3, 1),
                         (6, 64, 2, 2),
                         (6, 128, 1, 2)],
    "MobileNetSmallv1": [(1,  16, 1, 1),
                         (6,  24, 2, 2),
                         (6,  32, 3, 2),
                         (6,  64, 4, 1),
                         (6, 128, 2, 2),
                         (6, 256, 1, 1)],
    "MobileNetMediumv0": [(1,  16, 1, 1),
                         (5,  24, 2, 2),
                         (6,  32, 2, 2),
                         (6,  64, 3, 1),
                         (6,  96, 3, 1),
                         (5, 128, 2, 2),
                         (5, 256, 1, 2)]
}

maps_last_conv = {
    "MobileNetStandard": 1280,
    "MobileNetSmallv0": 512,
    "MobileNetSmallv1": 512,
    "MobileNetMediumv0": 1024
}

last_pool_size = { 
    # Recordar que se suele hacer el tamaño que llega, si llegan 10x10 -> 10
    "MobileNetStandard": 10,
    "MobileNetSmallv0": 5,
    "MobileNetSmallv1": 5, # Lo quiero dejar en 10 y hacer 2x2
    "MobileNetMediumv0": 5
}

flat_size = {
    "MobileNetStandard": 1280,
    "MobileNetSmallv0": 512,
    "MobileNetSmallv1": 512*2*2,
    "MobileNetMediumv0": 1024
}

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, ruido, dropout):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = apply_conv(in_planes, planes, kernel=(1,1), padding=0, stride=1, std=ruido, drop=dropout, act='relu', bn=True)
        self.conv2 = apply_conv(planes, planes, kernel=(3,3), padding=1, stride=stride, std=ruido, groups=planes, drop=dropout, act='relu', bn=True)
        self.conv3 = apply_conv(planes, out_planes, kernel=(1,1), padding=0, stride=1, std=ruido, drop=dropout, act='', bn=True)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):

    def __init__(self, mobilenet_name, dropout, ruido, gray, num_classes=2):
        super(MobileNetV2, self).__init__()
        self.name = mobilenet_name
        
        if gray: initial_channels = 1
        else: initial_channels = 3

        if dropout: self.conv1 = apply_conv(initial_channels, 32, kernel=(3,3), padding=1, stride=1, std=ruido, drop=0.1, act='relu', bn=True)
        else: self.conv1 = apply_conv(initial_channels, 32, kernel=(3,3), padding=1, stride=1, std=ruido, drop=0.0, act='relu', bn=True)
        self.layers = self._make_layers(mobilenet_name, ruido, dropout, in_planes=32)
        # cfg[mobilenet_name][-1][1] -> Los mapas de salida de make layers
        self.conv2 = apply_conv(cfg[mobilenet_name][-1][1], maps_last_conv[mobilenet_name], kernel=(1,1), padding=0, stride=1, std=ruido, drop=dropout, act='relu', bn=True)
        self.linear = nn.Linear(flat_size[mobilenet_name], num_classes)

    def _make_layers(self, mobilenet_name, ruido, dropout, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in cfg[mobilenet_name]:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, ruido, dropout))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layers(out)
        out = self.conv2(out)
        out = F.avg_pool2d(out, last_pool_size[self.name]) # Original 7x7 -> Tamaño salida
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out