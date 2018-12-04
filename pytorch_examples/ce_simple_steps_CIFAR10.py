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
import torchvision
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
import torch.nn.functional as F

try:
    from pytorchlib.pytorch_models import models_interface
    from pytorchlib.pytorch_library import utils_general, utils_training
except:
    print("Es necesario que pytorchlib este en mismo directorio!")


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
num_classes = 10
model_type, model_cfg_txt, optimizador = "VGG", "SmallVGG", "SGD"
model_cfg = [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M']
dropout = [0.1, 'M', 0.2, 'M', 0.3, 'M', 0.3, 'M', 0.4, 'M']
ruido = 0.0
input_channels = 3
flat_size = 512 # Tamaño del reshape en la VGG (si dejamos que corra el programa y es erroneo, no marcara cual es)
out_features = num_classes

growth_rate, last_pool_size, in_features = 0, 0, 0
out_type, block_type = "relu", ""
slack_channel = "general"

print("\nEntrenando CE Simple con {} - {} utilizando {} - CIFAR10!)".format(model_type, str(model_cfg_txt), optimizador))

# Establecemos una semilla para la replicacion de los experimentos correctamente
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

batch_size = 64

transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(3),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ])

trainset = torchvision.datasets.CIFAR10(root='~/DeepLearning/PytorchDatasets/CIFAR10', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='~/DeepLearning/PytorchDatasets/CIFAR10', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


""" ---- MODELOS ---- """

model = models_interface.select_model(model_type, model_config=model_cfg, dropout=dropout, ruido=ruido, input_channels=input_channels, growth_rate=growth_rate, flat_size=flat_size, in_features=in_features, out_type=out_type, block_type=block_type, out_features=out_features, last_pool_size=last_pool_size)

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
results["log-acc1"], results["log-acc3"] = [], []
results["time"] = ""

data_train_per_epoch = len(train_loader)
data_eval_per_epoch = len(test_loader)

""" ---- ENTRENAMIENTO DEL MODELO ---- """

# Las ultimas 50 las haremos con annealing lineal
epochs_steps = [75, 100, 100, 50, 15, 12]
lr_steps = [0.1, 0.01, 0.001, 0.001, 0.1, 0.01]
apply_lr_anneal_lineal = [False, False, False, True, False, True]
total_epochs = 1

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
            total_loss += utils_training.train_simple_model(model, batch_data, batch_target, loss_ce, model_optimizer, net_type="convolutional")
            if total_data_train >= data_train_per_epoch: break
            else: total_data_train += len(batch_data)

        acc1, acc3 = utils_training.evaluate_accuracy_models_generator([model], test_loader, max_data=data_eval_per_epoch, topk=(1,3), net_type="convolutional")
        curr_loss = total_loss / total_data_train
        print("Epoch {}: Learning Rate: {:.6f}, Train Loss: {:.6f}, acc@1: {:.2f}, acc@3: {:.2f} --- ".format(total_epochs, lr_new, curr_loss, acc1*100, acc3*100) + utils_general.time_to_human(start_time, time.time()))

        results["log-loss"].append(curr_loss)
        results["log-acc1"].append(acc1)
        results["log-acc3"].append(acc3)
        total_epochs+=1

        if acc1 > best_acc:
            best_acc = acc1
            best_model_state_dict = model.state_dict()
        
        # Decrementamos el learning rate solo para cuando vamos a hacer el ultimo set de epochs -> (indx+1) == len(epochs_steps)
        lr_new, model_optimizer = utils_training.anneal_lr_lineal([model], lr_now, epochs_now, epoch, optimizador, flag=apply_lr_anneal_lineal[indx])



print("CIFAR10 RESULTS - Acc@1 {:.2f}, Acc@3 {:.2f}".format(np.array(results["log-acc1"]).max()*100,np.array(results["log-acc3"]).max()*100))

""" ---- GUARDADO DE RESULTADOS Y LOGGING ---- """
results["time"] = utils_general.time_to_human(start_time, time.time())

pathlib.Path(model_name_path).mkdir(parents=True, exist_ok=True)
torch.save(best_model_state_dict, model_name_path+"CE_Simple_lrPlateau_checkpoint_state.pt")
with open(model_name_path+"CE_Simple_lrPlateau_LOG.pkl", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

utils_general.slack_message("(CE Simple - CIFAR10) Accuracy modelo " + model_type + " - " + str(model_cfg_txt) +
                            " utilizando " + optimizador +": " + str(np.max(np.array(results["log-acc1"])))[0:5] + "% --- " +
                            utils_general.time_to_human(start_time, time.time()), slack_channel)