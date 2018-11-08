''' Deep Residual Networks (ResNet) in Pytorch. '''
''' Official paper at https://arxiv.org/pdf/1512.03385.pdf '''
''' Original Implementation: https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from pytorchlib.pytorch_library import utils_nets

SHORTCUT_ADD_NAME = "ShortcutAdd"
SHORTCUT_JUMP_NAME = "ShortcutJump"

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
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, append2name=""):
        super(BasicBlock, self).__init__()

        self.block_forward = []
        self.block_forward.append(utils_nets.apply_conv(in_planes, planes, kernel=(3,3), stride=stride, padding=1, activation='relu', std=0.0, dropout=0.0, batchnorm=True, name_append=append2name))
        self.block_forward.append(utils_nets.apply_conv(planes, planes, kernel=(3,3), stride=1, padding=1, activation='linear', std=0.0, dropout=0.0, batchnorm=True, name_append=append2name))
        self.block_forward = nn.Sequential(*self.block_forward)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = utils_nets.apply_conv(in_planes, self.expansion*planes, kernel=1, stride=stride, padding=0, activation='linear', std=0.0, dropout=0.0, batchnorm=True, name_append="_"+SHORTCUT_JUMP_NAME+append2name)
        else:
            self.shortcut = nn.Sequential(OrderedDict([("_"+SHORTCUT_ADD_NAME+append2name, nn.Sequential())]))

    def forward(self, x):
        out = self.block_forward(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, append2name=""):
        super(Bottleneck, self).__init__()

        self.block_forward = []
        self.block_forward.append(utils_nets.apply_conv(in_planes, planes, kernel=1, stride=1, padding=1, activation='relu', std=0.0, dropout=0.0, batchnorm=True, name_append=append2name))
        self.block_forward.append(utils_nets.apply_conv(planes, planes, kernel=3, stride=stride, padding=1, activation='relu', std=0.0, dropout=0.0, batchnorm=True, name_append=append2name))
        self.block_forward.append(utils_nets.apply_conv(planes, self.expansion*planes, kernel=1, stride=1, padding=0, activation='linear', std=0.0, dropout=0.0, batchnorm=True, name_append=append2name))
        self.block_forward = nn.Sequential(*self.block_forward)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = utils_nets.apply_conv(in_planes, self.expansion*planes, kernel=1, stride=stride, padding=0, activation='linear', std=0.0, dropout=0.0, batchnorm=True, name_append="_"+SHORTCUT_JUMP_NAME+append2name)
        else:
            self.shortcut = nn.Sequential(OrderedDict([("_"+SHORTCUT_ADD_NAME+append2name, nn.Sequential())]))

    def forward(self, x):

        out = self.block_forward(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):

    def __init__(self, block, configuration_blocks, configuration_maps, gray, flat_size, num_classes, last_avg_pool_size=4):
        super(ResNet, self).__init__()
        self.in_planes = configuration_maps[0]
        self.last_avg_pool_size = last_avg_pool_size

        if gray: initial_channels = 1
        else: initial_channels = 3

        self.conv1 = nn.Conv2d(initial_channels, configuration_maps[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(configuration_maps[0])

        resnet_layers = []
        # La primera configuration_maps no la queremos ya que ya la hemos usado en conv1
        for indx, (channels, num_blocks) in enumerate(zip(configuration_maps[1:], configuration_blocks)):
            stride = 1 if indx == 0 else 2
            resnet_layers.append(self._make_layer(block, channels, num_blocks, stride=stride, append2name="_Block"+str(indx)))

        self.forward_conv = nn.Sequential(*resnet_layers)
        self.linear = nn.Sequential(OrderedDict([("FC_OUT", nn.Linear(flat_size, num_classes))]))

    def _make_layer(self, block, planes, num_blocks, stride, append2name):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, append2name=append2name))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.forward_conv(out)
        out = F.avg_pool2d(out, self.last_avg_pool_size)
        out = out.view(out.size(0), -1)
        try: out = self.linear(out)
        except: assert False, "The Flat size after view is: " + str(out.shape[1])
        return out


def ResNetModel(configuration_blocks, configuration_maps, block_type, gray, flat_size=0, num_classes=2):
    my_model = False
    if configuration_blocks in cfg_blocks:
        configuration_blocks = cfg_blocks[configuration_blocks]
    if configuration_maps in cfg_maps:
        configuration_maps = cfg_maps[configuration_maps]

    if "BasicBlock" in block_type: block_type = BasicBlock
    elif "Bottleneck" in block_type: block_type = Bottleneck
    else: assert False, "No block type '"+str(block_type)+"' allowed!"

    my_model = ResNet(block_type, configuration_blocks, configuration_maps, gray, flat_size, num_classes)
    my_model.net_type = "convolutional"
    return my_model


