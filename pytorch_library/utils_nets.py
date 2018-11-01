import torch
import torch.nn as nn
from collections import OrderedDict

LINEAR_NAME = "Linear"
CONV2D_NAME = "Conv2D"
BATCHNORM1D_NAME = "BatchNorm1D"
BATCHNORM2D_NAME = "BatchNorm2D"
GAUSSIANNOISE_NAME = "GaussianNoise"
DROPOUT1D_NAME = "Dropout1D"
DROPOUT1D_NAME = "Dropout2D"
MAXPOOL_NAME = "MAXPool"
AVGPOOL_NAME = "AVGPool"
ACT_RELU_NAME = "ACT_ReLU"
ACT_SOFTMAX_NAME = "ACT_Softmax"
ACT_LINEAR_NAME = "ACT_Linear"
ACT_TANH_NAME = "ACT_Tanh"
ACT_SIGMOID_NAME = "ACT_Sigmoid"

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#base class for all the models to allow for dynamic resize of tensor for noises
class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        std (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that std can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, std=0.1, is_relative_detach=True):
        super().__init__()
        self.std = std
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(DEVICE)

    def forward(self, x):
        if self.training and self.std != 0:
            scale = self.std * x.detach() if self.is_relative_detach else self.std * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x 


def get_activation(act, dim=1):
    if act == 'relu': return ACT_RELU_NAME, nn.ReLU()
    elif act == 'softmax': return ACT_SOFTMAX_NAME, nn.Softmax(dim)
    elif act == 'sigmoid': return ACT_SIGMOID_NAME, nn.Sigmoid()
    elif act == 'tanh': return ACT_TANH_NAME, nn.Tanh()
    # Con linear no modificamos la entrada
    # no hacemos nada === nn.Sequential()
    elif act == 'linear': return ACT_LINEAR_NAME, nn.Sequential()
    else: assert False, "Not valid activation function!"


def apply_linear(in_features, out_features, activation, std=0.0, dropout=0.0, batchnorm=True, name_append=""):

    # define the sequential
    use_bias = False if batchnorm else True
    forward_list = []
    forward_list.append((LINEAR_NAME + name_append, nn.Linear(in_features, out_features, bias=use_bias)))
    if batchnorm: forward_list.append((BATCHNORM1D_NAME + name_append, nn.BatchNorm1d(out_features)))
    if std != 0.0: forward_list.append((GAUSSIANNOISE_NAME + name_append, GaussianNoise(std=std)))
    name_activation, activation = get_activation(activation)
    forward_list.append((name_activation + name_append, activation))
    if dropout != 0.0: forward_list.append((DROPOUT1D_NAME + name_append, nn.Dropout(dropout)))
    return nn.Sequential(OrderedDict(forward_list))


def apply_conv(in_features, out_features, kernel=(3,3), activation="relu", std=0.0, dropout=0.0, batchnorm=True, stride=1, padding=1, name_append=""):
    # Diferencia entre extend y append https://stackoverflow.com/a/252711
    # define the sequential
    use_bias = False if batchnorm else True
    forward_list = []
    forward_list.append((CONV2D_NAME + name_append, nn.Conv2d(in_features, out_features, kernel, bias=use_bias, padding=padding, stride=stride)))
    if batchnorm: forward_list.append((BATCHNORM2D_NAME + name_append, nn.BatchNorm2d(out_features)))
    if std != 0.0: forward_list.append((GAUSSIANNOISE_NAME + name_append, GaussianNoise(std=std)))
    name_activation, activation = get_activation(activation)
    forward_list.append((name_activation + name_append, activation))
    if dropout != 0.0: forward_list.append((DROPOUT2D_NAME + name_append, nn.Dropout(dropout)))
    return nn.Sequential(OrderedDict(forward_list))


def apply_DeConv(in_features, out_features, kernel, activation, std=0.0, dropout=0.0, batchnorm=True, stride=1, padding=1, output_padding=0):
    # Diferencia entre extend y append https://stackoverflow.com/a/252711
    # define the sequential
    use_bias = False if batchnorm else True
    forward_list=[nn.ConvTranspose2d(in_features, out_features, kernel, stride=stride, padding=padding, bias=use_bias, output_padding=output_padding)]
    if batchnorm: forward_list.append(nn.BatchNorm2d(out_features))
    if std != 0.0: forward_list.append(GaussianNoise(std=std))
    forward_list.append(get_activation(activation))
    if drop!=0: forward_list.append(nn.Dropout2d(dropout))
    return nn.Sequential(*forward_list)


def apply_pool(pool_type, kernel_size, stride_size, name_append=""):
    pool_list = []
    if pool_type == "max_pool": pool_list.append((MAXPOOL_NAME + name_append, nn.MaxPool2d(kernel_size=kernel_size, stride=stride_size)))
    elif pool_type == "avg_pool": pool_list.append((AVGPOOL_NAME + name_append, nn.AvgPool2d(kernel_size=kernel_size, stride=stride_size)))    
    else: assert False, "Not valid pool!"
    return nn.Sequential(OrderedDict(pool_list))


def apply_DePool(kernel):
    return nn.UpsamplingBilinear2d(kernel)


class Identity(nn.Module):
    # Esta clase la usaremos para reemplazar las capas que deseemos por la identidad
    # Esto no hace nada y es como eliminar la capa deseada
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

def topk_classes(input, k):
    """
    Devuelve las k clases mas votadas de la entrada de mayor a menor probabilidad
    Para uso mas exhaustivo ir a https://pytorch.org/docs/stable/torch.html#torch.topk
    Ejemplo: si pasamos algo como [2, 4, 1, 8] con k=3 devolveria [3, 1, 0]
    """
    probs_values, class_indxs = torch.topk(input, k)
    return class_indxs

def models_average(outputs, scheme):
    """
    Dada una lista de salidas (outputs) promedia los resultados obtenidos
    teniendo dos posibildiades:
        - voting: donde la clase de mayor probabilidad vota 1 y el resto 0
        - sum: se suma la probabilidad de todas las clases para decidir
    """
    if not (type(outputs) is list or type(outputs) is tuple):
        assert False, "List of diferents outputs needed!"

    # Si no utilizamos el numpy() el tensor va por referencia y
    # modifica la entrada original
    if scheme == "sum":
        result = outputs[0].data.cpu().numpy()
        for output in outputs[1:]:
            result += output.data.cpu().numpy()
        return torch.from_numpy(result)
    elif scheme == "voting":
        # Es necesario tratar los outputs para pasarlos a codificacion 1 y 0s
        # https://discuss.pytorch.org/t/keep-the-max-value-of-the-array-and-0-the-others/14480/7
        view = outputs[0].view(-1, 5)
        result = (view == view.max(dim=1, keepdim=True)[0]).view_as(outputs[0])
        for output in outputs[1:]:
            view = output.view(-1, 5)
            result += (view == view.max(dim=1, keepdim=True)[0]).view_as(output)
        return result
    else: assert False, "Ivalid model average scheme!"