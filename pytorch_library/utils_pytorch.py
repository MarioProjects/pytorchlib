import torch
from torch import nn, optim
import os

def get_optimizer(optmizer_type, model_params, lr, pmomentum=0.9, pweight_decay=5e-4, palpha=0.9):
    # Funcion para rehacer el optmizador -> Ayuda para cambiar learning rate
    if optmizer_type=="SGD":
        return optim.SGD(filter(lambda p: p.requires_grad, model_params), lr=lr, momentum=pmomentum)
    elif optmizer_type=="Adam":
        return optim.Adam(filter(lambda p: p.requires_grad, model_params), lr=lr, weight_decay=pweight_decay)
    elif optmizer_type=="RMSprop":
        return optim.RMSprop(filter(lambda p: p.requires_grad, model_params), lr=lr, alpha=palpha)

    assert False, 'No optimizers with that name!'

def anneal_lr(lr_init, total_epochs, current_epoch):
    # Funcion para decrecer linealmente el learning rate
    lr_new = -(lr_init/total_epochs) * current_epoch + lr_init
    return lr_new

