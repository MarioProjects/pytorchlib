# coding: utf-8
""""""""""""""""""""""""""""""
""" Crossentropy Simple """
""""""""""""""""""""""""""""""

""" --- IMPORTACION LIBRERIAS --- """

import numpy as np
import os
import pandas as pd
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

from pytorchlib.pytorch_particular.quick_draw_doodle import data_generator as data_doodle
from pytorchlib.pytorch_particular.quick_draw_doodle import utils as utils_doodle

import albumentations


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Model train for Quick Draw Doodle Competition by Mario Parren~o. Msc student Universidad Politecnica de Valencia maparla@inf.upv.es . Enjoy!')
    parser.add_argument('--optimizer', type=str, choices=['SGD','Adam','RMSprop'], required=True, help='optimizer to apply')
    parser.add_argument('--model_type', type=str, required=True, default="", help='model type to use')
    parser.add_argument('--model_cfg', type=str, required=True, default="", help='model configuration to use')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch Size for training')
    parser.add_argument('--imgs_size', type=int, required=True, help='Size of images to work with')
    parser.add_argument('--flat_size', type=int, required=False, help='Input features to Fully-Connected part after convolutions')

    parser.add_argument('--block_type', type=str, default="", required=False, help='Block type for ResNet and SENet model')
    parser.add_argument('--last_pool_size', type=int, default=0, required=False, help='Last pool size of MobileNetv2 net')
    #parser.add_argument('--norm', type=str, choices=["zscore","0_1range","-1_1range", "np_range"], default="0_1range", help='Normalization to apply to the data')
    #parser.add_argument('--growth_rate', type=int, default=0, help='Growth rate for DenseNet models')
    aux=parser.parse_args()
    arguments=list()
    arguments=[aux.model_type, aux.model_cfg, aux.optimizer, aux.batch_size, aux.imgs_size, aux.flat_size, aux.block_type, aux.last_pool_size]
    return arguments

#model_type, model_cfg, optimizador, batch_size, size_imgs, flat_size, block_type, last_pool_size = parse_args()
model_type, model_cfg, optimizador, batch_size, size_imgs, flat_size, block_type, last_pool_size = "ResNet", "34_Small", "SGD", 2, 64, 1024, "BasicBlock", 10
data_amount = "All"
color = True
use_gray = not color

NCSVS = 100 # Numero de particiones de lo datos en ficheros de training
num_classes = 340
norm_data = "255"

print("\nEntrenando CE Simple con {} - {} utilizando {} - Quick Draw Doodle!)".format(model_type, str(model_cfg), optimizador))


""" ---- MODELOS ---- """

#model = models_interface.select_model(model_type, model_config="RESNET18", pretrained=True, out_features=num_classes).cuda()
#model = models_interface.select_model(model_type, model_config=model_cfg, pretrained=True, out_features=num_classes).cuda()
if "_" in model_cfg: model_config = model_cfg.split("_")
else: model_config = model_cfg
model = models_interface.select_model(model_type, model_config=model_config, flat_size=flat_size, block_type=block_type, gray=use_gray, out_features=num_classes, last_pool_size=last_pool_size)


""" --- CARGA DE DATOS --- """
# Establecemos una semilla para la replicacion de los experimentos correctamente
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)
np.random.seed(seed)

#train_transforms = [albumentations.HorizontalFlip(p=0.4),
#                    albumentations.ShiftScaleRotate(rotate_limit=6, border_mode=3, p=0.3)]
train_transforms = []
train_loader = data_doodle.image_generator_doodle(size=size_imgs, batch_size=batch_size, ks=range(NCSVS - 1),
                                                        data_amount=data_amount, norm=norm_data, transforms=train_transforms, color=color)

val_transforms = []
samples_validation = 3500
valid_df = pd.read_csv(os.path.join(data_doodle.SHUFFLED_CSVs_DIR + "50k" + "/", 'train_k{}.csv.gz'.format(NCSVS - 1)), nrows=samples_validation)
x_valid = data_doodle.df_to_image_array_doodle(valid_df, size_imgs, transforms=val_transforms, color=color)
y_valid = torch.from_numpy(np.array(valid_df.y))
valid_df = None # Limpiamos la variable


""" ---- CONSTANTES DE NUESTRO PROGRAMA ---- """

# Las ultimas 50 las haremos con annealing lineal
epochs_steps = [30, 20, 12, 10, 8, 10]
lr_steps = [0.35, 0.1, 0.01, 0.01, 0.1, 0.01]
apply_lr_anneal_lineal = [False, False, False, True, False, True]

# Vamos a utilizar la metrica del error cuadratico medio
loss_ce = nn.CrossEntropyLoss()
total_epochs = 1
best_acc = 0.0

model_name_path = "results/CE_Simple/"+optimizador+"/"+model_type+"/IMG"+str(size_imgs)+"/"+block_type+str(model_cfg)+"/"

results = {}
results["name"] = model_name_path
results["log-loss"] = []
results["log-acc1"] = []
results["log-acc3"] = []
results["time"] = ""

data_train_per_epoch = 200000

""" ---- ENTRENAMIENTO DEL MODELO ---- """

start_time = time.time()
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

        acc1, acc3 = utils_training.evaluate_accuracy_models_data([model], X_data=x_valid, y_data=y_valid, topk=(1,3))
        curr_loss = total_loss / total_data_train
        print("Epoch {}: Learning Rate: {:.4f}, Loss: {:.6f}, Accuracy1: {:.2f}, Accuracy3: {:.2f} --- ".format(total_epochs, lr_new, curr_loss, acc1*100, acc3*100) + utils_general.time_to_human(start_time, time.time()))

        results["log-loss"].append(curr_loss)
        results["log-acc1"].append(acc1)
        results["log-acc3"].append(acc3)
        total_epochs += 1

        if acc3 > best_acc:
            best_acc = acc3
            best_model_state_dict = model.state_dict()

        lr_new, model_optimizer = utils_training.anneal_lr([model], lr_now, epochs_now, epoch, optimizador, flag=apply_lr_anneal_lineal[indx])



""" ---- GUARDADO DE RESULTADOS Y LOGGING ---- """
results["time"] = utils_general.time_to_human(start_time, time.time())

pathlib.Path(model_name_path).mkdir(parents=True, exist_ok=True)
torch.save(best_model_state_dict, model_name_path+"/CE_Simple_checkpoint_state.pt")
with open(model_name_path+"/CE_Simple_LOG.pkl", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

utils_general.slack_message("(CE Simple - Quick Draw Doodle) Accuracy modelo " + model_type + " - " + str(model_cfg) +
                            " utilizando " + optimizador + ": acc@1 " + str(np.max(np.array(results["log-acc1"]))*100)[0:5] + "% " +
                            "acc@3 " + str(np.max(np.array(results["log-acc3"]))*100)[0:5] + "% --- " + utils_general.time_to_human(start_time, time.time()), "quick_draw_kaggle")


test_path = "/home/maparla/DeepLearning/KaggleDatasets/quick_draw_doodle/test_simplified.csv"
test_transforms = []
test_df = pd.read_csv(test_path)

x_test = data_doodle.df_to_image_array_doodle(test_df, size_imgs, transforms=test_transforms, norm=norm_data, color=color)

test_predictions = utils_training.predictions_models_data([model], x_test, batch_size=batch_size)
submission = utils_doodle.preds2submission(test_predictions, test_df)
submission.to_csv(model_name_path+'/NEW_SUBMISSION_CE_Simple_'+optimizador+'_'+model_type+"_"+block_type+str(model_cfg)+'_IMG'+str(size_imgs)+'.csv', index=False)