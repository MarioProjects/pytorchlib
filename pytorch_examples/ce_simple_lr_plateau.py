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
    -> LOAD YOUR DATA HERE!!!!!
    data_interface.database_selector()
"""

""" ---- MODELOS ---- """

model = models_interface.select_model(model_type, model_config=model_cfg, pretrained=True, out_features=num_classes).cuda()

""" ---- CONSTANTES DE NUESTRO PROGRAMA ---- """

# Vamos a utilizar la metrica del error cuadratico medio
loss_ce = nn.CrossEntropyLoss()
total_epochs = 100
lr_start = 0.1
best_acc = 0.0

model_name_path = "results/CE_Simple/"+optimizador+"/"+model_type+"/"+str(model_cfg_txt)+"/"

results = {}
results["name"] = model_name_path
results["log-loss"] = []
results["log-acc"] = []

data_train_per_epoch = train_samples
data_eval_per_epoch = val_samples

""" ---- ENTRENAMIENTO DEL MODELO ---- """

model_optimizer = utils_training.get_optimizer(optimizador, model.parameters(), lr=lr_start)
# https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'max', factor=0.5,
            patience=7, cooldown=3, threshold=0.005, min_lr=0, verbose=True) # Queremos maximizar el accuracy

for epoch in range(1, total_epochs+1):

    total_loss, total_data_train = 0, 0
    for batch_idx, data in enumerate(train_loader, 0):
        batch_data, batch_target = data
        batch_data = Variable(batch_data.cuda())
        batch_target = Variable(batch_target.cuda())
        total_loss += utils_training.train_simple_model(model, batch_data, batch_target, loss_ce, model_optimizer)
        if total_data_train >= data_train_per_epoch: break
        else: total_data_train += len(batch_data)

    curr_accuracy = utils_training.evaluate_accuracy_models_generator([model], val_loader, max_data=data_eval_per_epoch)
    curr_loss = total_loss / total_data_train
    print("Epoch {}: Learning Rate: {:.6f}, Loss: {:.6f}, Accuracy: {:.2f}".format(epoch, utils_training.get_current_lr(model_optimizer), curr_loss, curr_accuracy))

    results["log-loss"].append(curr_loss)
    results["log-acc"].append(curr_accuracy)

    if curr_accuracy > best_acc:
        best_acc = curr_accuracy
        best_model_state_dict = model.state_dict()

    scheduler.step(curr_accuracy)


""" ---- GUARDADO DE RESULTADOS Y LOGGING ---- """

pathlib.Path(model_name_path).mkdir(parents=True, exist_ok=True)
torch.save(best_model_state_dict, model_name_path+"CE_Simple_lrPlateau_checkpoint_state.pt")
with open(model_name_path+"CE_Simple_lrPlateau_LOG.pkl", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

utils_general.slack_message("(CE Simple - Quick Draw Doodle) Accuracy modelo " + model_type + " - " + str(model_cfg_txt) +
                            " utilizando " + optimizador +": " + str(np.max(np.array(results["log-acc"])))[0:5] + "%", slack_channel)