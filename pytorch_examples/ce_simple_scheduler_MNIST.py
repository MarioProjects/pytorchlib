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

try:
    from pytorchlib.pytorch_models import models_interface
    from pytorchlib.pytorch_library import utils_general, utils_training
except:
    assert False, "Es necesario que pytorchlib este en mismo directorio!"


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
model_type, model_cfg, model_cfg_txt, optimizador = "Simple_MLP", [256, 512, 512, num_classes, "|"], "256_512_512", "SGD"
dropout, ruido, input_channels = 0.0, 0.0, 0
growth_rate, flat_size, out_features, last_pool_size = 0, 0, 0, 0
out_type, block_type = "relu", ""
slack_channel = "general"

print("\nEntrenando CE Simple con {} - {} utilizando {} - MNIST!)".format(model_type, str(model_cfg_txt), optimizador))

# Establecemos una semilla para la replicacion de los experimentos correctamente
seed = 0
torch.manual_seed(seed=seed)
torch.cuda.manual_seed(seed=seed)

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/DeepLearning/PytorchDatasets/MNIST', train=True, download=True,
                    transform=transforms.Compose([
                        transforms.RandomRotation(4),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('~/DeepLearning/PytorchDatasets/MNIST', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=batch_size, shuffle=False)


in_features = 28*28*1 # Tamaño de las imagenes tras el crop en formato vector


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
results["log-loss-train"] = []
results["log-loss-val"] = []
results["log-acc1"], results["log-acc3"] = [], []
results["time"] = ""

data_train_per_epoch = len(train_loader)
data_eval_per_epoch = len(test_loader)

""" ---- ENTRENAMIENTO DEL MODELO ---- """

model_optimizer = utils_training.get_optimizer(optimizador, model.parameters(), lr=lr_start)
# https://pytorch.org/docs/stable/optim.html#torch.optim.lr_scheduler.ReduceLROnPlateau
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'max', factor=0.75,
            patience=8, cooldown=2, threshold=0.0005, min_lr=0.0005, verbose=True) # Queremos maximizar el accuracy


start_time = time.time()

for epoch in range(1, total_epochs+1):

    train_loss, total_data_train = 0, 0
    for batch_idx, data in enumerate(train_loader, 0):
        batch_data, batch_target = data
        batch_data = Variable(batch_data.cuda())
        batch_target = Variable(batch_target.cuda())
        train_loss += utils_training.train_simple_model(model, batch_data, batch_target, loss_ce, model_optimizer, net_type="fully-connected")
        if total_data_train >= data_train_per_epoch: break
        else: total_data_train += len(batch_data)

    acc1, acc3, val_loss = utils_training.evaluate_accuracy_loss_models_generator([model], test_loader, loss=loss_ce, max_data=data_eval_per_epoch, topk=(1,3), net_type="fully-connected")
    curr_loss = train_loss / total_data_train
    print("Epoch {}: Learning Rate: {:.6f}, Train Loss: {:.6f}, Test Loss: {:.6f}, acc@1: {:.2f}, acc@3: {:.2f} --- ".format(epoch, utils_training.get_current_lr(model_optimizer), train_loss, val_loss, acc1*100, acc3*100) + utils_general.time_to_human(start_time, time.time()))

    results["log-loss-train"].append(curr_loss)
    results["log-loss-val"].append(val_loss)
    results["log-acc1"].append(acc1)
    results["log-acc3"].append(acc3)

    if acc1 > best_acc:
        best_acc = acc1
        best_model_state_dict = model.state_dict()

    scheduler.step(val_loss)


print("MNIST RESULTS - Acc@1 {:.2f}, Acc@3 {:.2f}".format(np.array(results["log-acc1"]).max()*100,np.array(results["log-acc3"]).max()*100))

""" ---- GUARDADO DE RESULTADOS Y LOGGING ---- """
results["time"] = utils_general.time_to_human(start_time, time.time())

pathlib.Path(model_name_path).mkdir(parents=True, exist_ok=True)
torch.save(best_model_state_dict, model_name_path+"CE_Simple_lrPlateau_checkpoint_state.pt")
with open(model_name_path+"CE_Simple_lrPlateau_LOG.pkl", 'wb') as f:
    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

utils_general.slack_message("(CE Simple - MNIST) Accuracy modelo " + model_type + " - " + str(model_cfg_txt) +
                            " utilizando " + optimizador +": " + str(np.max(np.array(results["log-acc1"])))[0:5] + "% --- " +
                            utils_general.time_to_human(start_time, time.time()), slack_channel)