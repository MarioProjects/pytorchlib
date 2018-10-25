import torch
import os
from collections import OrderedDict
from pytorchlib.pytorch_models.vgg import VGGModel
from pytorchlib.pytorch_models.resnet import ResNetModel
from pytorchlib.pytorch_models.mobilenetv2 import MobileNetV2
from pytorchlib.pytorch_models.densenet import DenseNetModel
from pytorchlib.pytorch_models.basic_nets import BasicModel

def select_model(model_name, model_config=[], in_features=0, dropout=0, ruido=0, gray=0, growth_rate=0, out_type='relu', data_parallel=False):
    """ Model instantiator
    To instantiate models in a simple way
    Args:
        model_name: What kind of model you want to create. Example: "VGG", "ResNet"...
        model_config: To select the type of model to be instantiated, there are 
            preconfigured models and in some cases we can pass configuration lists 
            (look at the instantiators of each model for each case).
        in_features: How many input features enter in the model
        gray: If images were used in black and white (1) or not (0)
    Returns:
        An instance of the selected model
    """

    if 'VGG' in model_name:
        my_model = VGGModel(model_config, dropout, ruido, gray).cuda()
    elif 'ResNet' in model_name:
        my_model = ResNetModel(model_config, gray).cuda()
    elif 'DenseNet' in model_name:
        if not growth_rate: assert False, "Growth rate is required for DenseNets!"
        my_model = DenseNetModel(model_config, growth_rate, gray).cuda()
    elif 'MobileNet' in model_name:
        my_model = MobileNetv2Model(model_config, gray, num_classes).cuda()
    elif 'Simple_MLP' in model_name:
        my_model = BasicModel(model_config, "MLP", in_features, gray=gray, out_type=out_type).cuda()
    elif 'Simple_Convolutional' in model_name:
        my_model = BasicModel(model_config, "Convolutional", in_features, gray=gray, out_type=out_type).cuda()
    else: assert False, "Model '" + str(model_name) + "' not found!"

    if data_parallel: return torch.nn.DataParallel(my_model, device_ids=range(torch.cuda.device_count()))
    return my_model


def load_model(model_name, states_path, gray, dropout=0.0, ruido=0.0, growth_rate=0, in_features=0, out_type='relu', model_config=[]):

    if not os.path.exists(states_path): assert False, "Wrong Models States Path!"
    my_model = select_model(model_name, dropout, ruido, gray, growth_rate=growth_rate, in_features=in_features, out_type=out_type, model_config=model_config)
    model_state_dict = torch.load(states_path)

    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        if 'module' in k:
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else: 
            my_model.load_state_dict(model_state_dict)
            return my_model

    # load params
    my_model.load_state_dict(new_state_dict)
    return my_model