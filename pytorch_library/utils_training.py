import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable

def loss_fn_kd_kldivloss(outputs, teacher_outputs, labels, temperature, alpha=0.9):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    source: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py
    """
    alpha = alpha
    T = temperature
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss


def train_simple_model(model, data, target, loss, optimizer, out_pos=-1):
    # Losses: https://pytorch.org/docs/stable/nn.html
    model.train()
    optimizer.zero_grad()

    if model.net_type == "fully-connected":
        model_out = model.forward(Variable(data.view(data.shape[0], -1)))
    elif model.net_type == "convolutional":
        model_out = model.forward(Variable(data))

    # Algunos modelos devuelven varias salidas como pueden ser la capa
    # reshape y los logits, etc... Para conocer la salida a utilizar en el
    # loss lo que hacemos es tomar la que se indique en le parametro out_pos
    if type(model_out) is list or type(model_out) is tuple:
        model_out = model_out[out_pos]

    # Calculo el error obtenido
    cost = loss(model_out, target)
    cost.backward()

    # Actualizamos pesos y gradientes
    optimizer.step()

    return cost.item()


def evaluate_accuracy_models(models, data):

    for model in models:
        model.eval()

    correct_cnt_models, total_samples = [0]*len(models), 0
    for batch_idx, (batch, target) in enumerate(data):

        # calculo predicciones para el error de test de todos los modelos
        # Tengo que hacer el forward para cada modelo y ver que clases acierta
        for model_indx, model in enumerate(models):
            if model.net_type == "fully-connected":
                model_out = model.forward(Variable(batch.view(batch.shape[0], -1)))
            elif model.net_type == "convolutional":
                model_out = model.forward(Variable(batch))
            else: assert False, "Please define your model type!"

            # Algunos modelos devuelven varias salidas como pueden ser la capa
            # reshape y los logits, etc... Por lo que se establece el standar
            # de que la ultima salida sean los logits del modelo para hacer la clasificacion
            if type(model_out) is list or type(model_out) is tuple:
                model_out = model_out[-1]

            # Transformamos los logits a salida con el indice con el mayor valor
            #  de las tuplas que continen los logits
            _, pred_label = torch.max(model_out.data, 1)
            # sumo todas las que tengo bien, que tienen el valor que toca
            correct_cnt = (pred_label == target[:,0]).sum().item()
            correct_cnt_models[model_indx] += correct_cnt

        total_samples += batch.shape[0]

    accuracies = list(((np.array(correct_cnt_models) * 1.0) / total_samples)*100)
    if len(accuracies) == 1: return accuracies[0]
    return accuracies
