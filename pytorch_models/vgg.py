''' VGGs in Pytorch. '''
''' Official paper at https://arxiv.org/pdf/1409.1556.pdf '''
''' Original Implementation: https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py '''

import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from pytorchlib.pytorch_library import utils_nets

cfg = {
    'MNISTSmallVGG': [16, 'M', 16, 'M', 16, 'M', 32, 'M'], # Usado en MNIST (28,28) -> FlatSize 32*1*1

    # QUICK DRAW DOODLE::: Img Sizes: --> 32=FinalMaps*2*2, --> 64=FinalMaps*4*4...
    'QDrawSmallVGG': [32, 'M', 32, 'M', 64, 'M', 128, 'M'],
    'QDrawMediumVGG': [32, 'M', 64, 'M', 128, 'M', 256, 'M'],
    'QDrawLargeVGG': [32, 'M', 32, 'M', 64, 'M', 128, 'M', 256, 'M'],

    'ExtraSmallMiniVGGv0': [16, 'M', 16, 'M', 16, 'M', 32, 'M', 32, 'M'],
    'ExtraSmallMiniVGGv1': [16, 16, 'M', 16, 16, 'M', 16,16, 'M', 32,32, 'M', 32,32, 'M'],
    'SmallVGGv0': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'MediumVGGv0': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'BigVGGv0': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'StandardVGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'StandardVGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'StandardVGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'StandardVGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

dropout_values = {
    'ExtraSmallMiniVGGv0': [0.1, 'M', 0.2, 'M', 0.3, 'M', 0.4, 'M', 0.5, 'M'],
    'ExtraSmallMiniVGGv1': [0, 0.2, 'M', 0, 0.3, 'M', 0, 0.4, 'M', 0, 0.4, 'M', 0, 0.5, 'M'],
    'SmallVGGv0': [0.1, 'M', 0.2, 'M', 0.3, 'M', 0.4, 'M', 0.5, 'M'],
    'MediumVGGv0': [0, 0.2, 'M', 0, 0.3, 'M', 0, 0.4, 'M', 0, 0.4, 'M', 0, 0.5, 'M'],
    'BigVGGv0': [0, 0.2, 'M', 0, 0.3, 'M', 0, 0, 0.4, 'M', 0, 0, 0.4, 'M', 0, 0, 0.5, 'M'],
    'StandardVGG11': [0.1, 'M', 0.15, 'M', 0.2, 0.2, 'M', 0.35, 0.35, 'M', 0.45, 0.45, 'M'],
    'StandardVGG13': [0.1, 0.1, 'M', 0.15, 0.15, 'M', 0.2, 0.2, 'M', 0.35, 0.35, 'M', 0.45, 0.45, 'M'],
    'StandardVGG16': [0.1, 0.1, 'M', 0.15, 0.15, 'M', 0.2, 0.2, 0.2, 'M', 0.35, 0.35, 0.35, 'M', 0.45, 0.45, 0.45, 'M'],
    'StandardVGG19': [0.1, 0.1, 'M', 0.15, 0.15, 'M', 0.2, 0.2, 0.2, 0.2, 'M', 0.35, 0.35, 0.35, 0.35, 'M', 0.45, 0.45, 0.45, 0.45, 'M']
}


# El flat size pude variar dependiendo del tamaño de entrada de las imagenes
class VGG(nn.Module):
    def __init__(self, vgg_name, flat_size, dropout, ruido, gray, num_classes=2, custom_config=[]):
        super(VGG, self).__init__()

        if gray: in_channels = 1
        else: in_channels = 3

        conv_layers, num_layers = [], 0
        for indx, (channels) in enumerate(cfg[vgg_name]):

            last_layer = indx+1 == len(cfg[vgg_name]) or (indx+2 == len(cfg[vgg_name]) and cfg[vgg_name][-1] == "|")

            if channels == 'M':
                conv_layers.append(utils_nets.apply_pool("max_pool", 2, 2))
            else:
                if dropout and dropout_values[vgg_name][indx] != 0:
                    conv_layers.append(utils_nets.apply_conv(in_channels, channels, kernel=(3,3), activation='relu', std=ruido, padding=1, dropout=dropout_values[vgg_name][indx], batchnorm=True))
                else:
                    conv_layers.append(utils_nets.apply_conv(in_channels, channels, kernel=(3,3), activation='relu', std=ruido, padding=1, dropout=0.0, batchnorm=True))

                in_channels = channels
                num_layers += 1

        conv_layers.append(utils_nets.apply_pool("avg_pool", 1, 1, name_append="Reshape_OUT"))

        self.forward_conv = nn.Sequential(*conv_layers)
        self.forward_linear = layer = utils_nets.apply_linear(flat_size, num_classes, "linear", std=0.0,
                                                dropout=0.0, batchnorm=True, name_append="fc_OUT")


    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(x.size(0), -1)
        try: x = self.forward_linear(x)
        except: assert False, "The Flat size after view is: " + str(x.shape[1])

        return x


def VGGModel(vgg_name, flat_size, dropout, ruido, gray, num_classes=2):
    if vgg_name not in cfg:
        assert False, 'No VGG Model with that name!'
    else:
        my_model = VGG(vgg_name, flat_size, dropout, ruido, gray, num_classes=num_classes)
        return my_model
