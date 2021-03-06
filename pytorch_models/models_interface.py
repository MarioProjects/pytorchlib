'''
    https://pytorch.org/tutorials/beginner/saving_loading_models.html
'''

import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import os
from collections import OrderedDict
from pytorchlib.pytorch_models.vgg import VGGModel
from pytorchlib.pytorch_models.resnet import ResNetModel
from pytorchlib.pytorch_models.seresnext import SeResNeXtModel
from pytorchlib.pytorch_models.nasnet_a_large import NasNetALargeModel
from pytorchlib.pytorch_models.mobilenetv2 import MobileNetv2Model
from pytorchlib.pytorch_models.densenet import DenseNetModel
from pytorchlib.pytorch_models.basic_nets import BasicModel
from pytorchlib.pytorch_models.senet import SENetModel
from pytorchlib.pytorch_models.senet_pretrained import SENetPretrainedModel

def select_model(model_name, model_config=[], flat_size=0, in_features=0, out_features=0, dropout=0.0, ruido=0.0, input_channels=0, growth_rate=0, out_type='', batchnorm=True, default_act="relu", data_parallel=False, pretrained=False, block_type=None, last_pool_size=0, cardinality=32):
    """ Model instantiator
    To instantiate models in a simple way
    Args:
        model_name: What kind of model you want to create. Example: "VGG", "ResNet"...
        model_config: To select the type of model to be instantiated, there are
            preconfigured models and in some cases we can pass configuration lists
            (look at the instantiators of each model for each case).
        in_features: How many input features enter in the model
        input_channels: If images were used in black and white (1) or not (0)
        flat_size: size for the flatten/reshape in convolutional neural networks
    Returns:
        An instance of the selected model
    """

    if 'VGG' in model_name:
        my_model = VGGModel(model_config, flat_size, dropout, ruido, input_channels, num_classes=out_features).cuda()

    elif 'ResNet' in model_name:
        # Example: model = models_interface.select_model("ResNet", model_config=["Basic18","ExtraSmall"], flat_size=128*2*2, block_type="basic", input_channels=0, out_features=num_classes).cuda()
        # model_config[0] -> list of configuration_blocks or string with predefined option
        # model_config[1] -> list of configuration_maps or string with predefined option
        my_model = ResNetModel(model_config[0], model_config[1], block_type, input_channels, flat_size=flat_size, num_classes=out_features).cuda()

    elif 'DenseNet' in model_name:
        if not growth_rate: assert False, "Growth rate is required for DenseNets!"
        my_model = DenseNetModel(model_config, growth_rate, input_channels).cuda()

    elif 'MobileNetv2' in model_name:
        my_model = MobileNetv2Model(model_config, input_channels, out_features, flat_size, last_pool_size).cuda()

    elif 'SENet' in model_name:
        if pretrained:
            my_model = SENetPretrainedModel(model_config, out_features, "imagenet")
        else:
            my_model = SENetModel(model_config[0], model_config[1], block_type, input_channels, flat_size=flat_size, num_classes=out_features).cuda()
    
    elif 'SeResNeXt' in model_name:
        my_model = SeResNeXtModel(model_config[0], model_config[1], input_channels, block_type=block_type, flat_size=flat_size, num_classes=out_features, pretrained=pretrained).cuda()

    elif 'NASNetALarge' in model_name:
        my_model = NasNetALargeModel(input_channels, num_classes=out_features, pretrained=False).cuda()

    elif 'Simple_MLP' in model_name:
        my_model = BasicModel(model_config, "MLP", in_features, out_type, input_channels=input_channels, dropout=dropout, std=ruido, batchnorm=batchnorm, default_act=default_act).cuda()

    elif 'Simple_Convolutional' in model_name:
        my_model = BasicModel(model_config, "Convolutional", in_features, input_channels=input_channels, out_type=out_type).cuda()

    elif 'Imagenet' in model_name:
        # Example: model = models_interface.select_model("Imagenet", model_config="VGG11_BN", pretrained=True, out_features=num_classes).cuda()
        gray_transform = """You can transform input_channels images to fake 'RGB' with:
                            \ndata_transform = transforms.Compose([
                                \ttransforms.ToTensor(),
                                \ttransforms.Lambda(lambda x: torch.cat([x, x, x], 0))
                            ])\n"""

        if input_channels!=3: assert False, "ERROR: Imagenet models need to use color images! " + gray_transform

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

        if "INCEPTIONV3" == model_config: 
            my_model = models.inception_v3(pretrained=pretrained)
            my_model.fc = nn.Linear(2048, out_features)
        else:
            # Si queremos reentrenar un modelo reemplazamos la ultima capa de salida
            if out_features:

                # Here, we need to freeze all the network except the final layer.
                # We need to set requires_grad == False to freeze the parameters
                # so that the gradients are not computed in backward().
                for param in my_model.parameters():
                    param.requires_grad = False

                # Parameters of newly constructed modules have requires_grad=True by default
                if hasattr(my_model, 'classifier'):
                    if len(my_model.classifier._modules)!=0:
                        num_ftrs = my_model.classifier._modules[str(len(my_model.classifier._modules)-1)].in_features
                        my_model.classifier._modules[str(len(my_model.classifier._modules)-1)] = nn.Linear(num_ftrs, out_features)
                    elif len(my_model.classifier._modules)==0:
                        num_ftrs = my_model.classifier.in_features
                        my_model.classifier = nn.Linear(num_ftrs, out_features)
                    else: assert False, "Check the my_model last linear!"
                elif hasattr(my_model, 'fc'):
                    if len(my_model.fc._modules)!=0:
                        num_ftrs = my_model.fc._modules[str(len(my_model.fc._modules)-1)].in_features
                        my_model.fc._modules[str(len(my_model.fc._modules)-1)] = nn.Linear(num_ftrs, out_features)
                    elif len(my_model.fc._modules)==0:
                        num_ftrs = my_model.fc.in_features
                        my_model.fc = nn.Linear(num_ftrs, out_features)
                    else: assert False, "Check the my_model last linear!"
                else: assert False, "Check the my_model last linear!"

        # https://pytorch.org/docs/stable/torchvision/models.html
        print("""\nWARNING: The images (3, 224,244) have to be loaded in to a range
                of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406]
                and std = [0.229, 0.224, 0.225]\n""")

    else: assert False, "Model '" + str(model_name) + "' not found!"

    if data_parallel: return torch.nn.DataParallel(my_model, device_ids=range(torch.cuda.device_count()))
    return my_model.cuda()


def load_model(model_name, model_config=[], states_path="", model_path="", input_channels=0, pretrained=False, dropout=0.0, ruido=0.0, growth_rate=0, in_features=0, flat_size=0, out_features=0, out_type='relu', block_type=None, last_pool_size=0, cardinality=32, data_parallel=False):

    if model_path!="" and os.path.exists(model_path):
        return torch.load(model_path)
    elif model_path != "": assert False, "Wrong Model Path!"

    if not os.path.exists(states_path): assert False, "Wrong Models_States Path!"

    my_model = select_model(model_name, model_config=model_config, dropout=dropout, ruido=ruido, pretrained=pretrained, input_channels=input_channels, growth_rate=growth_rate, flat_size=flat_size, in_features=in_features, out_type=out_type, block_type=block_type, out_features=out_features, last_pool_size=last_pool_size, cardinality=cardinality, data_parallel=data_parallel)
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