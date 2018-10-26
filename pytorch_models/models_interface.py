import torch
from torchvision import datasets, models, transforms
import os
from collections import OrderedDict
from pytorchlib.pytorch_models.vgg import VGGModel
from pytorchlib.pytorch_models.resnet import ResNetModel
from pytorchlib.pytorch_models.mobilenetv2 import MobileNetv2Model
from pytorchlib.pytorch_models.densenet import DenseNetModel
from pytorchlib.pytorch_models.basic_nets import BasicModel

def select_model(model_name, model_config=[], in_features=0, out_features=0, dropout=0.0, ruido=0.0, gray=0, growth_rate=0, out_type='relu', data_parallel=False, pretrained=True):
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
        my_model = BasicModel(model_config, "MLP", in_features, gray=gray, out_type=out_type, dropout=dropout, std=ruido).cuda()

    elif 'Simple_Convolutional' in model_name:
        my_model = BasicModel(model_config, "Convolutional", in_features, gray=gray, out_type=out_type).cuda()

    elif 'Imagenet' in model_name:
        # https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
        if "VGG11" == model_config: my_model = models.vgg11(pretrained=pretrained)
        if "VGG11_BN" == model_config: my_model = models.vgg11_bn(pretrained=pretrained)
        if "VGG13" == model_config: my_model = models.vgg13(pretrained=pretrained)
        if "VGG13_BN" == model_config: my_model = models.vgg13_bn(pretrained=pretrained)
        if "VGG16" == model_config: my_model = models.vgg16(pretrained=pretrained)
        if "VGG16_BN" == model_config: my_model = models.vgg16_bn(pretrained=pretrained)
        if "VGG19" == model_config: my_model = models.vgg19(pretrained=pretrained)
        if "VGG19_BN" == model_config: my_model = models.vgg19_bn(pretrained=pretrained)

        if "RESNET18" == model_config: my_model = models.resnet18(pretrained=pretrained)
        if "RESNET34" == model_config: my_model = models.resnet34(pretrained=pretrained)
        if "RESNET50" == model_config: my_model = models.resnet50(pretrained=pretrained)
        if "RESNET101" == model_config: my_model = models.resnet101(pretrained=pretrained)
        if "RESNET152" == model_config: my_model = models.resnet152(pretrained=pretrained)

        if "DENSENET121" == model_config: my_model = models.densenet121(pretrained=pretrained)
        if "DENSENET169" == model_config: my_model = models.densenet169(pretrained=pretrained)
        if "DENSENET161" == model_config: my_model = models.densenet161(pretrained=pretrained)
        if "DENSENET201" == model_config: my_model = models.densenet201(pretrained=pretrained)

        my_model.net_type = "convolutional"
        # Si queremos reentrenar un modelo reemplazamos la ultima capa de salida
        if out_features:
            num_ftrs = my_model.fc.in_features
            my_model.fc = nn.Linear(num_ftrs, out_features)

        # https://pytorch.org/docs/stable/torchvision/models.html
        print("""\nWARNING: The images (3, 224,244) have to be loaded in to a range
                of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
                and std = [0.229, 0.224, 0.225]\n""")

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