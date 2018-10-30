import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable


def get_optimizer(optmizer_type, model_params, lr, pmomentum=0.9, pweight_decay=5e-4, palpha=0.9):
    # Funcion para rehacer el optmizador -> Ayuda para cambiar learning rate
    if optmizer_type=="SGD":
        return optim.SGD(filter(lambda p: p.requires_grad, model_params), lr=lr, momentum=pmomentum)
    elif optmizer_type=="Adam":
        return optim.Adam(filter(lambda p: p.requires_grad, model_params), lr=lr, weight_decay=pweight_decay)
    elif optmizer_type=="RMSprop":
        return optim.RMSprop(filter(lambda p: p.requires_grad, model_params), lr=lr, alpha=palpha)

    assert False, 'No optimizers with that name!'

def anneal_lr(redes, lr_init, total_epochs, current_epoch, optimizer_type, flag=True):
    # flag nos indica si realmente queremos hacer el annel sobre las redes
    if not flag: lr_new = lr_init
    else: lr_new = -(lr_init/total_epochs) * current_epoch + lr_init

    redes_resultado = []
    for red in redes:
        redes_resultado.append(get_optimizer(optimizer_type, red.parameters(), lr_new))
    if len(redes_resultado) == 1: return lr_new, redes_resultado[0]
    return lr_new, redes_resultado


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


def simple_target_creator(samples, value):
    """
    Funcion para crear un vector utilizado para asignar la clase de las
    diferentes muestras durante el entrenamiento de tamaño 'samples'
    El vector sera de (samplesx1) lleno de 'value'
    """
    return Variable(torch.ones(samples, 1)).type(torch.cuda.FloatTensor)*value


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


def evaluate_accuracy_models(models, data, max_data=0):

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
        if max_data != 0 and max_data >= total_samples: break

    accuracies = list(((np.array(correct_cnt_models) * 1.0) / total_samples)*100)
    if len(accuracies) == 1: return accuracies[0]
    return accuracies


def train_discriminator(discriminator_net, discriminator_optimizer, real_data, fake_data, loss):
    num_samples = real_data.size(0) # Para conocer el numero de muestras

    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    
    # 1.1 ----> Train with real
    # Reseteamos los gradientes
    discriminator_optimizer.zero_grad()
    discriminator_net.zero_grad()
    # prediction on Real Data
    prediction_real = discriminator_net(real_data)
    # Calculate error and backpropagate
    # Debemos tener en cuenta que son reales -> 1s
    error_real = loss(prediction_real, simple_target_creator(num_samples, 1))
    error_real.backward()

    # 1.2 ----> Train on Fake Data
    prediction_fake = discriminator_net(fake_data)
    # Calculate error and backpropagate
    # Debemos tener en cuenta que son falsos -> 0s
    error_fake = loss(prediction_fake, simple_target_creator(num_samples, 0))
    error_fake.backward()

    # 1.3 Update weights with gradients of discriminator
    discriminator_optimizer.step()

    # Return error
    return error_real.item() + error_fake.item()


def train_generator(discriminator_net, generator_optimizer, fake_data, loss):
    num_samples = fake_data.size(0) # Para conocer el numero de muestras

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    # Reseteamos gradientes
    generator_optimizer.zero_grad()

    # Inferimos nuestros datos falsos a traves del discriminador para
    # posteriormente tratar de 'engañarlo'
    prediction = discriminator_net(fake_data)

    # Calculate error and backpropagate
    # IMPORTANTE -> Queremos que el generador aprenda a que
    # sus muestras sean clasificadas como reales, por lo que
    # CALCULAMOS EL LOSS CON 1s! como si fueran reales
    error = loss(prediction, simple_target_creator(num_samples, 1))
    error.backward()

    # 3. Actualizamos pesos y gradientes del generador
    generator_optimizer.step()

    # Return error
    return error.item()