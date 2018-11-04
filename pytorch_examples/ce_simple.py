# coding: utf-8
""""""""""""""""""""""""""
""" Crossentropy Simple """
""""""""""""""""""""""""""

""" --- IMPORTACION LIBRERIAS --- """

import numpy as np
import pickle
import pathlib

import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F

from pytorchlib.pytorch_models import models_interface
from pytorchlib.pytorch_data import data_interface
from pytorchlib.pytorch_library import utils_general, utils_training


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Model train for Quick Draw Doodle Competition by Mario Parren~o. Msc student Universidad Politecnica de Valencia maparla@inf.upv.es . Enjoy!')
    parser.add_argument('--optimizer', type=str, choices=['SGD','Adam','RMSprop'], required=True, help='optimizer to apply')
    parser.add_argument('--model_type', type=str, required=True, default="", help='model type to use')
    parser.add_argument('--model_cfg', type=str, required=True, default="", help='model configuration to use')
    #parser.add_argument('--batch_size', type=int, default=128, help='Batch Size for training')
    #parser.add_argument('--norm', type=str, choices=["zscore","0_1range","-1_1range", "np_range"], default="0_1range", help='Normalization to apply to the data')
    #parser.add_argument('--growth_rate', type=int, default=0, help='Growth rate for DenseNet models')
    aux=parser.parse_args()
    arguments=list()
    arguments=[aux.model_type, aux.model_cfg, aux.optimizer]
    return arguments

# model_type, model_cfg, optimizador = parse_args()
model_type, model_cfg, optimizador = "Imagenet", "RESNET50", "SGD"
slack_channel = "quick_draw_kaggle"
num_classes = 2

if type(model_cfg)==list or type(model_cfg)==tuple:
    model_cfg_txt = '_'.join(model_cfg)
else: model_cfg_txt = model_cfg

print("\nEntrenando CE Simple con {} - {} utilizando {} - Quick Draw Doodle!)".format(model_type, str(model_cfg_txt), optimizador))

""" --- CARGA DE DATOS --- """
# Establecemos una semilla para la replicacion de los experimentos correctamente
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

batch_size = 64

"""
train_samples, train_loader, = data_interface.database_selector("QuickDraw", seed=0,
                                                 data_segment="0_10000", batch_size=batch_size,
                                                 norm='', desired_img_size=(224, 224),
                                                 evaluation_mode=False)

val_samples, val_loader, = data_interface.database_selector("QuickDraw", seed=0,
                                                 data_segment="50000_52500", batch_size=batch_size,
                                                 norm='', desired_img_size=(224, 224),
                                                 evaluation_mode=True)
"""

""" ---- MODELOS ---- """

#model = models_interface.select_model(model_type, model_config="RESNET18", pretrained=True, out_features=num_classes).cuda()
model = models_interface.select_model(model_type, model_config=model_cfg, flat_size=32*2*2, out_features=num_classes).cuda()

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

data_train_per_epoch = 5000
data_eval_per_epoch = 1000

""" ---- ENTRENAMIENTO DEL MODELO ---- """

for indx,(epochs_now, lr_now) in enumerate(zip(epochs_steps, lr_steps)):

    model_optimizer = utils_training.get_optimizer(optimizador, model.parameters(), lr_now)
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

        curr_accuracy = utils_training.evaluate_accuracy_models([model], val_loader, max_data=data_eval_per_epoch)
        curr_loss = total_loss / total_data_train
        print("Epoch {}: Learning Rate: {:.4f}, Loss: {:.6f}, Accuracy: {:.2f}".format(total_epochs, lr_new, curr_loss, curr_accuracy))

        results["log-loss"].append(curr_loss)
        results["log-acc"].append(curr_accuracy)
        total_epochs += 1

        if curr_accuracy > best_acc:
            best_acc = curr_accuracy
            best_model_state_dict = model.state_dict()

        # Decrementamos el learning rate solo para cuando vamos a hacer el ultimo set de epochs -> (indx+1) == len(epochs_steps)
        lr_new, model_optimizer = utils_training.anneal_lr([model], lr_now, epochs_now, epoch, optimizador, flag=apply_lr_anneal_lineal[indx])


""" ---- GUARDADO DE RESULTADOS Y LOGGING ---- """

pathlib.Path(model_name_path).mkdir(parents=True, exist_ok=True)
torch.save(best_model_state_dict, model_name_path+"CE_Simple_checkpoint_state.pt")
with open(model_name_path+"CE_Simple_LOG.pkl", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

utils_general.slack_message("(CE Simple - Quick Draw Doodle) Accuracy modelo " + model_type + " - " + str(model_cfg_txt) +
                            " utilizando " + optimizador +": " + str(np.max(np.array(results["log-acc"])))[0:5] + "%", slack_channel)