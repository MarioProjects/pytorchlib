import types
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def get_optimizer(optmizer_type, model_params, lr=0.1, pmomentum=0.9, pweight_decay=5e-4, palpha=0.9):
    # Funcion para rehacer el optmizador -> Ayuda para cambiar learning rate
    if optmizer_type=="SGD":
        return optim.SGD(filter(lambda p: p.requires_grad, model_params), lr=lr, momentum=pmomentum)
    elif optmizer_type=="Adam":
        return optim.Adam(filter(lambda p: p.requires_grad, model_params), lr=lr, weight_decay=pweight_decay)
    elif optmizer_type=="RMSprop":
        return optim.RMSprop(filter(lambda p: p.requires_grad, model_params), lr=lr, alpha=palpha)

    assert False, 'No optimizers with that name!'

def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def anneal_lr(redes, lr_init, total_epochs, current_epoch, optimizer_type, flag=True):
    # flag nos indica si realmente queremos hacer el annel sobre las redes
    if not flag: lr_new = lr_init
    else: lr_new = -(lr_init/total_epochs) * current_epoch + lr_init

    redes_resultado = []
    for red in redes:
        redes_resultado.append(get_optimizer(optimizer_type, red.parameters(), lr=lr_new))
    if len(redes_resultado) == 1: return lr_new, redes_resultado[0]
    return lr_new, redes_resultado


def defrost_model_params(model):
    # Funcion para descongelar redes!
    for param in model.parameters():
        param.requires_grad = True


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
        model_out = model.forward(Variable(data.float().view(data.shape[0], -1)))
    elif model.net_type == "convolutional":
        model_out = model.forward(Variable(data.float()))

    # Algunos modelos devuelven varias salidas como pueden ser la capa
    # reshape y los logits, etc... Para conocer la salida a utilizar en el
    # loss lo que hacemos es tomar la que se indique en le parametro out_pos
    if type(model_out) is list or type(model_out) is tuple:
        model_out = model_out[out_pos]

    # Calculo el error obtenido
    # Cuidado con la codificacion one hot! https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/8
    try: cost = loss(model_out, target)
    except: cost = loss(model_out, target[:,0])
    cost.backward()

    # Actualizamos pesos y gradientes
    optimizer.step()

    return cost.item()


def evaluate_accuracy_models_generator(models, data, max_data=0, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Si paso un modelo y topk(1,5) -> acc1, acc5,
    # Si paso dos modelo y topk(1,5) -> m1_acc1, m1_acc5, m2_acc1, m2_acc5
    with torch.no_grad():

        if type(topk)==int: 
            maxk = topk
            topk = (topk,)
        else: maxk = max(topk)

        correct_models, total_samples = [0]*len(models), 0
        for batch_idx, (batch, target) in enumerate(data):

            batch_size = target.size(0)

            # calculo predicciones para el error de test de todos los modelos
            # Tengo que hacer el forward para cada modelo y ver que clases acierta
            for model_indx, model in enumerate(models):
                if model.net_type == "fully-connected":
                    model_out = model.forward(Variable(batch.float().view(batch.shape[0], -1).cuda()))
                elif model.net_type == "convolutional":
                    model_out = model.forward(Variable(batch.float().cuda()))
                else: assert False, "Please define your model type!"

                # Algunos modelos devuelven varias salidas como pueden ser la capa
                # reshape y los logits, etc... Por lo que se establece el standar
                # de que la ultima salida sean los logits del modelo para hacer la clasificacion
                if type(model_out) is list or type(model_out) is tuple:
                    model_out = model_out[-1]

                # Transformamos los logits a salida con el indice con el mayor valor
                #  de las tuplas que continen los logits
                _, pred = model_out.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.cuda().view(1, -1).expand_as(pred.cuda()))

                res_topk = []
                for k in topk:
                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    res_topk.append(correct_k.mul_(100.0))
                res_topk = np.array(res_topk)

                correct_models[model_indx] += res_topk

            total_samples += batch_size
            if max_data != 0 and total_samples >= max_data: break

    accuracies = []
    for result_model in correct_models:
        for topkres in result_model:
            accuracies.append((topkres*1.0)/total_samples)

    #accuracies = list(((np.array(correct_models) * 1.0) / total_samples))
    if len(accuracies) == 1: return accuracies[0]
    return accuracies

def evaluate_accuracy_models_data(models, X_data, y_data, batch_size=100, max_data=0, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Si paso un modelo y topk(1,5) -> acc1, acc5,
    # Si paso dos modelo y topk(1,5) -> m1_acc1, m1_acc5, m2_acc1, m2_acc5
    with torch.no_grad():

        if type(topk)==int: 
            maxk = topk
            topk = (topk,)
        else: maxk = max(topk)

        correct_models, total_samples = [0]*len(models), 0

        total_samples = 0
        while True: 

            # Debemos comprobar que no nos pasamos con el batch_size
            if total_samples + batch_size >= len(X_data): batch_size = (len(X_data)-1) - total_samples

            batch = X_data[total_samples:total_samples+batch_size]
            target = y_data[total_samples:total_samples+batch_size]

            # calculo predicciones para el error de test de todos los modelos
            # Tengo que hacer el forward para cada modelo y ver que clases acierta
            for model_indx, model in enumerate(models):
                if model.net_type == "fully-connected":
                    model_out = model.forward(Variable(batch.float().view(batch.shape[0], -1).cuda()))
                elif model.net_type == "convolutional":
                    model_out = model.forward(Variable(batch.float().cuda()))
                else: assert False, "Please define your model type!"

                # Algunos modelos devuelven varias salidas como pueden ser la capa
                # reshape y los logits, etc... Por lo que se establece el standar
                # de que la ultima salida sean los logits del modelo para hacer la clasificacion
                if type(model_out) is list or type(model_out) is tuple:
                    model_out = model_out[-1]

                # Transformamos los logits a salida con el indice con el mayor valor
                #  de las tuplas que continen los logits
                _, pred = model_out.topk(maxk, 1, True, True)
                pred = pred.t()
                correct = pred.eq(target.cuda().view(1, -1).expand_as(pred.cuda()))

                res_topk = []
                for k in topk:
                    correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                    res_topk.append(correct_k.mul_(100.0))
                res_topk = np.array(res_topk)

                correct_models[model_indx] += res_topk

            total_samples+=batch_size
            if max_data != 0 and total_samples >= max_data or total_samples+1 == len(X_data): break

    accuracies = []
    for result_model in correct_models:
        for topkres in result_model:
            accuracies.append((topkres*1.0)/total_samples)

    #accuracies = list(((np.array(correct_models) * 1.0) / total_samples))
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