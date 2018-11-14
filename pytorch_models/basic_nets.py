''' MiMic Network '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from pytorchlib.pytorch_library import utils_nets

MLP_CONFIGURATIONS = {
    'MLPSmallMiMic': [512, 1024, 2048, 512, "|", 2, "|"],
    'MLPMediumMiMic': [1024, 2048, 4096, 512, "|", 2, "|"],
    'MLPDecreasingMiMic': [4096, 2048, 1024, 512, "|", 2, "|"],
    
    'MLPSmallMiMicReshape': [512, 1024, 2048, 512, "|"],
    'MLPMediumMiMicReshape': [1024, 2048, 4096, 512, "|"],
    'MLPDecreasingMiMicReshape': [4096, 2048, 1024, 512, "|"],

    'MLPSmallMiMicLogits': [512, 1024, 2048, 512, 2, "|"],
    'MLPMediumMiMicLogits': [1024, 2048, 4096, 512, 2, "|"],
    'MLPDecreasingMiMicLogits': [4096, 2048, 1024, 512, 2, "|"]
}

class MLPNet(nn.Module):
    """ To create MLPs
    MultiLayer Perceptron generator
    Args:
        mlp_cfg: The configuration of MLP. Introduce "|" after
        each layer you want know the output. Example [128, 256, num_clases, "|"]
        in_features: How many input features enter in the model
        out_type: Desired output type ("relu" or "linear" implemented)
    Returns:
        A Module that fits your mlp_cfg
    """
    def __init__(self, mlp_cfg, in_features, out_type, dropout=0.0, std=0.0, batchnorm=True, default_act="relu"):
        super(MLPNet, self).__init__()

        # Check if choosed prebuilt configuration
        if type(mlp_cfg) is str: mlp_cfg = MLP_CONFIGURATIONS[mlp_cfg]

        linear_layers, self.get_output, num_layers = [], [], 0
        for indx, (n_features) in enumerate(mlp_cfg):
            if n_features == "|": self.get_output[-1] = True
            else:
                self.get_output.append(False)
                last_layer = indx+1 == len(mlp_cfg) or (indx+2 == len(mlp_cfg) and mlp_cfg[-1] == "|")

                """
                - You can access the components through their names as follows:
                    for forward_lineal in nlpnetmodel.children():
                        for sequential in child.children():
                            for operation in sequential.named_children():
                                # named_children is a list (name, children_component)
                                name = operation[0]
                                parameter = operation[1]
                                if "OUT" in name: parameter.requires_grad = True
                                else: parameter.requires_grad = False
                """

                if last_layer:  default_act, dropout, std = out_type, 0.0, 0.0
                layer = utils_nets.apply_linear(in_features, n_features, default_act, std=std, 
                                                dropout=dropout, batchnorm=batchnorm)

                linear_layers.append(layer)
                # Cambiamos las ultimas neuronas de entrada para poder conectar las capas correctamente
                in_features = n_features
                num_layers += 1

        self.forward_linear = nn.Sequential(*linear_layers)

    def forward(self, x):
        out = []
        for indx, (step) in enumerate(self.forward_linear.children()):
            # Iterate over Sequential modules created
            # Si nos encontramos con una barra "|" es porque
            # queremos obtener el valor de aplicar la anterior Sequential
            # Si no aplicamos el step (Sequential) que toque

            # Por algun tipo de problema no puedo hacer step(x) asi que itero sobre las
            # componentes internas del step (Linear-BatchNorm-ReLU) y voy operando
            #for step_component in step.children():
            x = step(x)

            if self.get_output[indx]: out.append(x)

        if len(out) == 0: assert False, "MLP not returns nothing?"
        elif len(out) == 1: return out[0]
        return out


class ConvNet(nn.Module):
    """
    Red convolucional de tres capas ocultas que tratará de aprender
    la representación intermedia que se obtinene en un paso
    de una red convolucional a traves del error cuadratico medio
    """
    def __init__(self, out_type='relu', dropout=0.0, std=0.0):
        super(ConvNet, self).__init__()

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
        
        #try: out = self.linear(out)
        #except: assert False, "The Flat size after view is: " + str(out.shape[1])
        
        return reshape, logits


def BasicModel(model_cfg, model_type, in_features, out_type, gray=0, dropout=0.0, std=0.0, batchnorm=True, default_act="relu"):
    if "MLP" in model_type:
        return MLPNet(model_cfg, in_features, out_type, dropout=dropout, std=std, batchnorm=batchnorm, default_act=default_act)
    elif model_cfg == "ConvMiMic":
        return ConvNet(out_type, dropout=dropout, std=std, batchnorm=batchnorm, default_act=default_act)
    elif type(model_cfg) is list or type(model_cfg) is tuple:
        if "Convolutional" in model_type:
            return ConvNet(out_type, dropout=dropout, std=std, batchnorm=batchnorm, default_act=default_act)
    assert False, 'No Basic Networks networks with this name!'