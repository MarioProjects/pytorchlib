''' VGGs in Pytorch. '''
''' Official paper at https://arxiv.org/pdf/1409.1556.pdf '''
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import Linear_act,return_activation,apply_conv,apply_linear,apply_pool, MamasitaNetwork,add_gaussian,apply_DePool,apply_DeConv

num_classes = 2

cfg = {
    'ExtraSmallMiniVGGv0': [16, 'M', 16, 'M', 16, 'M', 32, 'M', 32, 'M'],
    'ExtraSmallMiniVGGv1': [16, 16, 'M', 16, 16, 'M', 16,16, 'M', 32,32, 'M', 32,32, 'M'],
    'SmallVGGv0': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'],
    'MediumVGGv0': [32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'BigVGGv0': [32, 32, 'M', 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
}

dropout_values = {
    'ExtraSmallMiniVGGv0': [0.1, 'M', 0.2, 'M', 0.3, 'M', 0.4, 'M', 0.5, 'M'],
    'ExtraSmallMiniVGGv1': [0.0, 0.2, 'M', 0.0, 0.3, 'M', 0.0, 0.4, 'M', 0.0, 0.4, 'M', 0.0, 0.5, 'M'],
    'SmallVGGv0': [0.1, 'M', 0.2, 'M', 0.3, 'M', 0.4, 'M', 0.5, 'M'],
    'MediumVGGv0': [0.0, 0.2, 'M', 0.0, 0.3, 'M', 0.0, 0.4, 'M', 0.0, 0.4, 'M', 0.0, 0.5, 'M'],
    'BigVGGv0': [0.0, 0.2, 'M', 0.0, 0.3, 'M', 0.0, 0.0, 0.4, 'M', 0.0, 0.0, 0.4, 'M', 0.0, 0.0, 0.5, 'M'],
}

flat_size = {
    'ExtraSmallMiniVGGv0': 32*2*2,
    'ExtraSmallMiniVGGv1': 32*2*2,
    'SmallVGGv0': 512*2*2,
    'MediumVGGv0': 512*2*2,
    'BigVGGv0': 512*2*2,
}

"""
-- Original --
'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
"""

class MamasitaNetwork(nn.Module):
    #utils
    dynamicgeneration=[0]
    dynamicbatch=[0]

class VGG(MamasitaNetwork):
    def __init__(self, vgg_name, dropout, ruido, gray):
        super(VGG, self).__init__()
        self.forward_conv = self._make_layers(vgg_name, ruido, dropout, gray)


        if 'Mini' in vgg_name:
            lin1=apply_linear(flat_size[vgg_name], 32,'relu', drop=dropout, bn=True)
            lin2=apply_linear(32,16,'relu', drop=dropout, bn=True)
            lin3=apply_linear(16, num_classes, 'linear', bn=True)
            self.forward_linear = nn.Sequential(lin1,lin2,lin3)
        else:
            lin1=apply_linear(flat_size[vgg_name], 512,'relu', drop=dropout, bn=True)
            lin2=apply_linear(512,256,'relu', drop=dropout, bn=True)
            lin3=apply_linear(256,256,'relu', drop=dropout, bn=True)
            lin4=apply_linear(256, num_classes, 'linear', bn=True)
            self.forward_linear = nn.Sequential(lin1,lin2,lin3,lin4)
        self.vgg_name = vgg_name

        # Con reduce=False lo que hacemos es que no se sume
        # para todos los batches y se haga la media del loss de forma automatica
        self.cost_func=torch.nn.CrossEntropyLoss(reduce=False)

    def forward(self, x):
        x = self.forward_conv(x)
        x = x.view(-1, flat_size[self.vgg_name])
        x = self.forward_linear(x)
        return x

    def _make_layers(self, vgg_name, std, dropout, gray):
        layers = []

        if gray: in_channels = 1
        else: in_channels = 3

        for indx, (layer) in enumerate(cfg[vgg_name]):
            if layer == 'M':
                layers += [apply_pool(kernel=(2,2))]
            else:
                # SOLO le voy a poner Dropout a la ultima convolucional de cada 'bloque'
                if dropout!=0.0:
                    layers += [apply_conv(in_channels, layer, kernel=(3,3), act='relu', std=std, drop=dropout_values[vgg_name][indx], bn=True)]
                else:
                    layers += [apply_conv(in_channels, layer, kernel=(3,3), act='relu', std=std, drop=0.0, bn=True)]
                in_channels = layer
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGGModel(vgg_name, dropout, ruido, gray):
    if vgg_name not in cfg:
        assert False, 'No VGG Model with that name!'
    else:
        my_model = VGG(model_name, dropout, ruido, gray)
        my_model.net_type = "convolutional"
        return my_model
