''' MiMic Network '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

cfg = {
    'SmallMiMic': [512, 1024, 2048],
    'MediumMiMic': [1024, 2048, 4096],
    'DecreasingMiMic': [4096, 2048, 1024]
}

class MLPNet(nn.Module):
    """
    Red neuronal de tres capas ocultas que tratará de aprender
    la representación intermedia que se obtinene en un paso 
    de una red convolucional a traves del error cuadratico medio
    """
    def __init__(self, which_cfg, in_features, out_features, out_type='relu'):
        super(MLPNet, self).__init__()
        self.net_type = "fully-connected"
        
        n_features = in_features # Imagen de entrada en vector (Imagen de 80x80)
        n_out = out_features # Numero neuronas que aprenderemos

        # Primera capa oculta, entran n_features y tenemos 1024 neuronas
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, cfg[which_cfg][0]),
            nn.BatchNorm1d(cfg[which_cfg][0]),
            nn.ReLU()
        )
        
        # Segunda capa oculta, entran 1024 neuronas de la capa anterior y salen 2048
        self.hidden1 = nn.Sequential(
            nn.Linear(cfg[which_cfg][0], cfg[which_cfg][1]),
            nn.BatchNorm1d(cfg[which_cfg][1]),
            nn.ReLU()
        )
        
        # Tercera capa oculta, entran 2048 neuronas de la capa anterior y salen 4096
        self.hidden2 = nn.Sequential(
            nn.Linear(cfg[which_cfg][1], cfg[which_cfg][2]),
            nn.BatchNorm1d(cfg[which_cfg][2]),
            nn.ReLU()            
        )
        
        # Capa de salida, entran 4096 neuronas de la capa anterior y sacamos las neuronas deseadas
        if out_type == 'relu':
            self.reshapeLayer = nn.Sequential(
                nn.Linear(cfg[which_cfg][2], n_out),
                nn.BatchNorm1d(n_out),
                nn.ReLU()
            )
            self.out = nn.Sequential()

        elif out_type == 'linear':
            # La salida linear la vamos a querer cuando queremos aprender
            # los logits y tendremos por lo general 512 neuronas previas 
            # representando el reshape
            self.reshapeLayer = nn.Sequential(
                nn.Linear(cfg[which_cfg][2], 512),
                nn.BatchNorm1d(512),
                nn.ReLU()
            )

            self.out = nn.Sequential(
                nn.Linear(512, n_out),
                nn.BatchNorm1d(n_out)
            )

        else: assert False, 'No out type valid for this MiMic network!'

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        reshape = self.reshapeLayer(x)
        logits = self.out(reshape)
        return reshape, logits


class ConvNet(nn.Module):
    """
    Red convolucional de tres capas ocultas que tratará de aprender
    la representación intermedia que se obtinene en un paso 
    de una red convolucional a traves del error cuadratico medio
    """
    def __init__(self, out_type='relu'):
        super(ConvNet, self).__init__()
        self.net_type = "convolutional"

        self.num_channels = 8

        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
        # stride, padding). We also include batch normalisation layers that help stabilise training.
        # For more details on how to use these layers, check out the documentation.
        self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.num_channels)

        self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.num_channels*2)

        self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.num_channels*4)

        # 2 fully connected layers to transform the output of the convolution layers to the final output
        self.fc1 = nn.Linear(4*4*self.num_channels*4, 2)


    def forward(self, x):
        #                                                  -> batch_size x 3 x 80 x 80
        # we apply the convolution layers, followed by batch normalisation, maxpool and relu
        x = self.bn1(self.conv1(x)) #                       -> batch_size x num_channels x 80 x 80
        x = F.relu(F.max_pool2d(x, 2)) #                    -> batch_size x num_channels x 40 x 40

        x = self.bn2(self.conv2(x)) #                       -> batch_size x num_channels x 40 x 40
        x = F.relu(F.max_pool2d(x, 2)) #                    -> batch_size x num_channels x 20 x 20

        x = self.bn3(self.conv3(x)) #                       -> batch_size x num_channels x 20 x 20
        x = F.relu(F.max_pool2d(x, 5)) #                   -> batch_size x num_channels x 4 x 4

        # flatten the output for each image
        reshape = x.view(-1, 4*4*self.num_channels*4)             # batch_size x 4*4*num_channels*4
        logits = self.fc1(reshape)
        return reshape, logits


def BasicModel(model_name, in_features, out_features, out_type="relu"):
    if model_name in ["SmallMiMic", "MediumMiMic", "DecreasingMiMic"]:
        return MLPNet(model_name, in_features, out_features, out_type)
    elif model_name == "ConvMiMic":
        return ConvNet(out_type)
    assert False, 'No Basic Networks networks with this name!'