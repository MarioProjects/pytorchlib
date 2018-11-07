'''SENet in PyTorch.
SENet is the winner of ImageNet-2017. The paper is not released yet.
Original Implementation: https://github.com/kuangliu/pytorch-cifar/blob/master/models/senet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from pytorchlib.pytorch_library import utils_nets

cfg_blocks = {
    '18': [2,2,2,2],
    '34': [3,4,6,3],
    '50': [3,4,6,3],
    '101': [3,4,23,3],
    '152': [3,8,36,3],
}

cfg_maps = {
    'ExtraSmall': [16, 16, 32, 64, 128],
    'Small': [32, 32, 64, 128, 256],
    'Standard': [64, 64, 128, 256, 512]
}


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.0, std=0.0, append2name=""):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        se_layers = []
        se_layers.append(utils_nets.apply_conv(planes, planes//16, activation="relu", std=std, kernel=1,
                                                dropout=dropout, batchnorm=False, name_append=append2name))
        se_layers.append(utils_nets.apply_conv(planes//16, planes, activation="sigmoid", std=std, kernel=1,
                                                dropout=dropout, batchnorm=False, name_append=append2name))
        self.se_operation = nn.Sequential(*se_layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = self.se_operation(w)

        # Excitation
        out = out * w  # New broadcasting feature from v0.2!

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, dropout=0.0, std=0.0, append2name=""):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        se_layers = []
        se_layers.append(utils_nets.apply_conv(planes, planes//16, activation="relu", std=std, kernel=1,
                                                dropout=dropout, batchnorm=False, name_append=append2name))
        se_layers.append(utils_nets.apply_conv(planes//16, planes, activation="sigmoid", std=std, kernel=1,
                                                dropout=dropout, batchnorm=False, name_append=append2name))
        self.se_operation = nn.Sequential(*se_layers)


    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = self.se_operation(w)

        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):

    def __init__(self, block, configuration_blocks, configuration_maps, gray, flat_size, num_classes, last_avg_pool_size=4):
        super(SENet, self).__init__()
        self.in_planes = configuration_maps[0]
        self.last_avg_pool_size = last_avg_pool_size

        if gray: initial_channels = 1
        else: initial_channels = 3


        self.conv1 = nn.Conv2d(initial_channels, configuration_maps[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(configuration_maps[0])

        senet_layers = []
        # La primera configuration_maps no la queremos ya que ya la hemos usado en conv1
        for indx, (channels, num_blocks) in enumerate(zip(configuration_maps[1:], configuration_blocks)):
            stride = 1 if indx == 0 else 2
            senet_layers.append(self._make_layer(block, channels, num_blocks, stride, append2name="_Block"+str(indx)))

        self.forward_conv = nn.Sequential(*senet_layers)
        self.linear = nn.Sequential(OrderedDict([("FC_OUT", nn.Linear(flat_size, num_classes))]))


    def _make_layer(self, block, planes, num_blocks, stride, append2name=""):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride=stride, append2name=append2name))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.forward_conv(out)
        out = F.avg_pool2d(out, self.last_avg_pool_size)
        out = out.view(out.size(0), -1)
        try: out = self.linear(out)
        except: assert False, "The Flat size after view is: " + str(out.shape[1])
        return out


def SENetModel(configuration_blocks, configuration_maps, block_type, gray, flat_size=0, num_classes=2):
    my_model = False
    if configuration_blocks in cfg_blocks:
        configuration_blocks = cfg_blocks[configuration_blocks]
    if configuration_maps in cfg_maps:
        configuration_maps = cfg_maps[configuration_maps]

    if "BasicBlock" in block_type: block_type = BasicBlock
    elif "PreActBlock" in block_type: block_type = PreActBlock
    else: assert False, "Not block type allowed!"

    my_model = SENet(block_type, configuration_blocks, configuration_maps, gray, flat_size, num_classes)
    my_model.net_type = "convolutional"
    return my_model
