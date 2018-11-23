# coding: utf-8
""""""""""""""""""""""""""
""" Crossentropy Simple """
""""""""""""""""""""""""""

""" --- IMPORTACION LIBRERIAS --- """

import numpy as np
import pickle
import pathlib
import time

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F

from pytorchlib.pytorch_models import models_interface
from pytorchlib.pytorch_library import utils_general, utils_training

import albumentations
from pytorchlib.pytorch_particular.vgg_face2 import data_generator
import torch.utils.data as torchdata

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Model train for Quick Draw Doodle Competition by Mario Parren~o. Msc student Universidad Politecnica de Valencia maparla@inf.upv.es . Enjoy!')
    parser.add_argument('--optimizer', type=str, choices=['SGD','Adam','RMSprop'], default="SGD", help='optimizer to apply')
    parser.add_argument('--model_type', type=str, required=True, default="", help='model type to use')
    parser.add_argument('--model_cfg', type=str, required=True, default="", help='model configuration to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for training')
    parser.add_argument('--flat_size', type=int, default=0, help='Flat Size for model')

    parser.add_argument('--block_type', type=str, default="", required=False, help='Block type for ResNet and SENet model')
    parser.add_argument('--last_pool_size', type=int, default=0, required=False, help='Last pool size of MobileNetv2 net')
    #parser.add_argument('--norm', type=str, choices=["zscore","0_1range","-1_1range", "np_range"], default="0_1range", help='Normalization to apply to the data')
    #parser.add_argument('--growth_rate', type=int, default=0, help='Growth rate for DenseNet models')
    aux=parser.parse_args()
    arguments=list()
    arguments=[aux.model_type, aux.model_cfg, aux.optimizer, aux.batch_size, aux.flat_size, aux.block_type, aux.last_pool_size]
    return arguments


#model_type, model_cfg, optimizador, batch_size, flat_size, block_type, last_pool_size = parse_args()
model_type, model_cfg, optimizador, batch_size, flat_size, block_type, last_pool_size = "MobileNetv2", "MobileNetSmallv0", "SGD", 32, 512, "", 5
dropout, ruido, growth_rate, gray = 0.0, 0.0, 0, 0
in_features, out_type = 0, ""
norm = "0_1range"
num_classes = 2
slack_channel = "log_ai_work"

if type(model_cfg)==list or type(model_cfg)==tuple:
    model_cfg_txt = '_'.join(model_cfg)
else: model_cfg_txt = model_cfg

print("\nEntrenando CE Simple con {} - {} utilizando {} - VGGFaces2!)".format(model_type, str(model_cfg_txt), optimizador))


# -> LOAD YOUR DATA HERE!!!!!
# Establecemos una semilla para la replicacion de los experimentos correctamente
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

transforms_train = [albumentations.RandomCrop(80, 80), albumentations.HorizontalFlip(p=0.4), albumentations.Rotate(limit=5, p=0.3)]
train_data_path = '/home/maparla/DeepLearning/PytorchDatasets/VGGFace2/train_gender/'
train_dataset = data_generator.FoldersDatasetVGGFace2(data_path=train_data_path, transforms=transforms_train, normalization=norm)
train_loader=torchdata.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

transforms_test = [albumentations.CenterCrop(80, 80)]
test_data_path = '/home/maparla/DeepLearning/PytorchDatasets/VGGFace2/test_gender/'
test_dataset = data_generator.FoldersDatasetVGGFace2(data_path=test_data_path, transforms=transforms_test, normalization=norm)
test_loader=torchdata.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


""" ---- MODELOS ---- """

model = models_interface.select_model(model_type, model_config=model_cfg, dropout=dropout, ruido=ruido, gray=gray, growth_rate=growth_rate, flat_size=flat_size, in_features=in_features, out_type=out_type, block_type=block_type, out_features=num_classes, last_pool_size=last_pool_size)

""" ---- CONSTANTES DE NUESTRO PROGRAMA ---- """

# Las ultimas 50 las haremos con annealing lineal
epochs_steps = [50, 100, 100, 50]
lr_steps = [0.1, 0.01, 0.001, 0.001]
apply_lr_anneal_lineal = [False, False, False, True]

# Vamos a utilizar la metrica del error cuadratico medio
loss_ce = nn.CrossEntropyLoss()
total_epochs = 1
best_acc = 0.0

model_name_path = "results/CE_Simple/"+optimizador+"/"+model_type+"/"+str(model_cfg_txt)+"/"

results = {}
results["name"] = model_name_path
results["log-loss"] = []
results["log-acc"] = []
results["time"] = ""

data_train_per_epoch = len(train_dataset)
data_eval_per_epoch = len(test_dataset)

""" ---- ENTRENAMIENTO DEL MODELO ---- """

start_time = time.time()
for indx,(epochs_now, lr_now) in enumerate(zip(epochs_steps, lr_steps)):

    model_optimizer = utils_training.get_optimizer(optimizador, model.parameters(), lr=lr_now)
    lr_new = lr_now

    for epoch in range(epochs_now):

        total_loss, total_data_train = 0, 0
        for batch_idx, data in enumerate(train_loader, 0):
            batch_data, batch_target = data
            batch_data = Variable(batch_data.cuda())
            batch_target = Variable(batch_target.cuda())
            total_loss += utils_training.train_simple_model(model, batch_data, batch_target, loss_ce, model_optimizer)
            if total_data_train >= data_train_per_epoch: break
            else: total_data_train += len(batch_data)

        acc1 = utils_training.evaluate_accuracy_models_generator([model], test_loader, max_data=data_eval_per_epoch, topk=(1,))
        curr_loss = total_loss / total_data_train
        print("Epoch {}: Learning Rate: {:.6f}, Loss: {:.6f}, Accuracy: {:.2f} --- ".format(total_epochs, lr_new, curr_loss, acc1*100) + utils_general.time2human(start_time, time.time()))

        results["log-loss"].append(curr_loss)
        results["log-acc"].append(acc1)
        total_epochs += 1

        if acc1 > best_acc:
            best_acc = acc1
            best_model_state_dict = model.state_dict()

        # Decrementamos el learning rate solo para cuando vamos a hacer el ultimo set de epochs -> (indx+1) == len(epochs_steps)
        lr_new, model_optimizer = utils_training.anneal_lr([model], lr_now, epochs_now, epoch, optimizador, flag=apply_lr_anneal_lineal[indx])


""" ---- GUARDADO DE RESULTADOS Y LOGGING ---- """
results["time"] = utils_general.time2human(start_time, time.time())

pathlib.Path(model_name_path).mkdir(parents=True, exist_ok=True)
torch.save(best_model_state_dict, model_name_path+"CE_Simple_checkpoint_state.pt")
with open(model_name_path+"CE_Simple_LOG.pkl", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

utils_general.slack_message("(CE Simple - VGGFaces2) Accuracy modelo " + model_type + " - " + str(model_cfg_txt) +
                            " utilizando " + optimizador +": " + str(np.max(np.array(results["log-acc"])))[0:5] + "% --- " +
                            utils_general.time2human(start_time, time.time()), slack_channel)