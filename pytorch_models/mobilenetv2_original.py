''' MobileNetV2 in PyTorch. '''
''' Oficial paper at https://arxiv.org/abs/1801.04381 '''

import torch
import torch.nn as nn
import torch.nn.functional as F

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
                         (6,  32, 3, 2),
                         (6,  64, 4, 1),
                         (6, 128, 2, 2),
                         (6, 256, 1, 2)],
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
                         (6,  96, 2, 1),
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
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):

    def __init__(self, mobilenet_name, num_classes=2):
        super(MobileNetV2, self).__init__()
        self.name = mobilenet_name
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(mobilenet_name, in_planes=32)
        # cfg[mobilenet_name][-1][1] -> Los mapas de salida de make layers
        self.conv2 = nn.Conv2d(cfg[mobilenet_name][-1][1], maps_last_conv[mobilenet_name], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(maps_last_conv[mobilenet_name])
        self.linear = nn.Linear(flat_size[mobilenet_name], num_classes)

    def _make_layers(self, mobilenet_name, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in cfg[mobilenet_name]:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, last_pool_size[self.name]) # Original 7x7 -> Tamaño salida
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out