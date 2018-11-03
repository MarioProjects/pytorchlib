import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import os
from collections import OrderedDict
from pytorchlib.pytorch_models.vgg import VGGModel
from pytorchlib.pytorch_models.resnet import ResNetModel
from pytorchlib.pytorch_models.mobilenetv2 import MobileNetv2Model
from pytorchlib.pytorch_models.densenet import DenseNetModel
from pytorchlib.pytorch_models.basic_nets import BasicModel

def select_model(model_name, model_config=[], flat_size=0, in_features=0, out_features=0, dropout=0.0, ruido=0.0, gray=0, growth_rate=0, out_type='', batchnorm=True, default_act="relu", data_parallel=False, pretrained=True, block_type=None):
    """ Model instantiator
    To instantiate models in a simple way
    Args:
        model_name: What kind of model you want to create. Example: "VGG", "ResNet"...
        model_config: To select the type of model to be instantiated, there are
            preconfigured models and in some cases we can pass configuration lists
            (look at the instantiators of each model for each case).
        in_features: How many input features enter in the model
        gray: If images were used in black and white (1) or not (0)
        flat_size: size for the flatten/reshape in convolutional neural networks
    Returns:
        An instance of the selected model
    """

    if 'VGG' in model_name:
        my_model = VGGModel(model_config, flat_size, dropout, ruido, gray, num_classes=out_features).cuda()

    elif 'ResNet' in model_name:
        # Example: model = models_interface.select_model("ResNet", model_config=["Basic18","ExtraSmall"], flat_size=128*2*2, block_type="basic", gray=0, out_features=num_classes).cuda()
        # model_config[0] -> list of configuration_blocks or string with predefined option
        # model_config[1] -> list of configuration_maps or string with predefined option
        my_model = ResNetModel(model_config[0], model_config[1], block_type, gray, flat_size=flat_size, num_classes=out_features).cuda()

    elif 'DenseNet' in model_name:
        if not growth_rate: assert False, "Growth rate is required for DenseNets!"
        my_model = DenseNetModel(model_config, growth_rate, gray).cuda()

    elif 'MobileNet' in model_name:
        my_model = MobileNetv2Model(model_config, gray, num_classes).cuda()

    elif 'Simple_MLP' in model_name:
        my_model = BasicModel(model_config, "MLP", in_features, out_type, gray=gray, dropout=dropout, std=ruido, batchnorm=batchnorm, default_act=default_act).cuda()

    elif 'Simple_Convolutional' in model_name:
        my_model = BasicModel(model_config, "Convolutional", in_features, gray=gray, out_type=out_type).cuda()

    elif 'Imagenet' in model_name:
        # Example: model = models_interface.select_model("Imagenet", model_config="VGG11_BN", pretrained=True, out_features=num_classes).cuda()
        gray_transform = """You can transform gray images to fake 'RGB' with:
                            \ndata_transform = transforms.Compose([
                                \ttransforms.ToTensor(),
                                \ttransforms.Lambda(lambda x: torch.cat([x, x, x], 0))
                            ])\n"""

        if gray: assert False, "ERROR: Imagenet models need to use color images! " + gray_transform

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

            # Here, we need to freeze all the network except the final layer.
            # We need to set requires_grad == False to freeze the parameters
            # so that the gradients are not computed in backward().
            for param in my_model.parameters():
                param.requires_grad = False

            # Parameters of newly constructed modules have requires_grad=True by default
            if hasattr(model, 'classifier'):
                if len(model.classifier._modules)!=0:
                    num_ftrs = model.classifier._modules[str(len(model.classifier._modules)-1)].in_features
                    model.classifier._modules[str(len(model.classifier._modules)-1)] = nn.Linear(num_ftrs, out_features)
                elif len(model.classifier._modules)==0:
                    num_ftrs = model.classifier.in_features
                    model.classifier = nn.Linear(num_ftrs, out_features)
                else: assert False, "Check the model last linear!"
            elif hasattr(model, 'fc'):
                if len(model.fc._modules)!=0:
                    num_ftrs = model.fc._modules[str(len(model.fc._modules)-1)].in_features
                    model.fc._modules[str(len(model.fc._modules)-1)] = nn.Linear(num_ftrs, out_features)
                elif len(model.fc._modules)==0:
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, out_features)
                else: assert False, "Check the model last linear!"
            else: assert False, "Check the model last linear!"

        # https://pytorch.org/docs/stable/torchvision/models.html
        print("""\nWARNING: The images (3, 224,244) have to be loaded in to a range
                of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
                and std = [0.229, 0.224, 0.225]\n""")

    else: assert False, "Model '" + str(model_name) + "' not found!"

    if data_parallel: return torch.nn.DataParallel(my_model, device_ids=range(torch.cuda.device_count()))
    return my_model


def load_model(model_name, model_config=[], states_path="", gray=0, dropout=0.0, ruido=0.0, growth_rate=0, in_features=0, flat_size=0, out_features=0, out_type='relu', block_type=None):

    if not os.path.exists(states_path): assert False, "Wrong Models_States Path!"
    my_model = select_model(model_name, model_config=model_config, dropout=dropout, ruido=ruido, gray=gray, growth_rate=growth_rate, flat_size=flat_size, in_features=in_features, out_type=out_type, block_type=block_type, out_features=out_features)
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