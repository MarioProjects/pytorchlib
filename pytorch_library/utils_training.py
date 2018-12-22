import types
import numpy as np
import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable

CROSS_ENTROPY_ONE_HOT_WARNING = False

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


def anneal_lr_lineal(models, lr_init, total_epochs, current_epoch, optimizer_type, flag=True):
    # flag nos indica si realmente queremos hacer el annel sobre las models
    if not flag: lr_new = lr_init
    else: lr_new = -(lr_init/total_epochs) * current_epoch + lr_init

    redes_resultado = []
    for model in models:
        redes_resultado.append(get_optimizer(optimizer_type, model.parameters(), lr=lr_new))
    if len(redes_resultado) == 1: return lr_new, redes_resultado[0]
    return lr_new, redes_resultado


def defrost_model_params(model):
    # Funcion para descongelar redes!
    for param in model.parameters():
        param.requires_grad = True


def simple_target_creator(samples, value):
    """
    Funcion para crear un vector utilizado para asignar la clase de las
    diferentes muestras durante el entrenamiento de tamaño 'samples'
    El vector sera de (samplesx1) lleno de 'value'
    """
    return Variable(torch.ones(samples, 1)).type(torch.cuda.FloatTensor)*value


def train_simple_model(model, data, target, loss, optimizer, out_pos=-1, target_one_hot=False, net_type="convolutional", do_step=True):
    # Losses: https://pytorch.org/docs/stable/nn.html
    if(model.training==False): model.train()

    if net_type == "fully-connected":
        model_out = model.forward(Variable(data.float().view(data.shape[0], -1)))
    elif net_type == "convolutional":
        model_out = model.forward(Variable(data.float()))

    # Algunos modelos devuelven varias salidas como pueden ser la capa
    # reshape y los logits, etc... Para conocer la salida a utilizar en el
    # loss lo que hacemos es tomar la que se indique en le parametro out_pos
    if type(model_out) is list or type(model_out) is tuple:
        model_out = model_out[out_pos]

    if target_one_hot: _, target = target.max(dim=1)

    # Calculo el error obtenido
    # Cuidado con la codificacion one hot! https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/8
    try: cost = loss(model_out, target)
    except:
        global CROSS_ENTROPY_ONE_HOT_WARNING
        if not CROSS_ENTROPY_ONE_HOT_WARNING:
            print("\nWARNING-INFO: Crossentropy not works with one hot target encoding!\n")
            CROSS_ENTROPY_ONE_HOT_WARNING = True
        cost = loss(model_out, target[:,0])
    cost.backward()

    if do_step:
        # Actualizamos pesos y gradientes
        optimizer.step()
        optimizer.zero_grad()

    return cost.item()


def evaluate_accuracy_models_generator(models, data, max_data=0, topk=(1,), target_one_hot=False, net_type="convolutional"):
    """Computes the accuracy (sobre 1) over the k top predictions for the specified values of k"""
    # Si paso un modelo y topk(1,5) -> acc1, acc5,
    # Si paso dos modelo y topk(1,5) -> m1_acc1, m1_acc5, m2_acc1, m2_acc5
    with torch.no_grad():

        if type(topk)==int:
            maxk = topk
            topk = (topk,)
        else: maxk = max(topk)

        correct_models, total_samples = [0]*len(models), 0
        for batch_idx, (batch, target) in enumerate(data):

            if target_one_hot: _, target = target.max(dim=1)
            batch_size = target.size(0)

            # calculo predicciones para el error de test de todos los modelos
            # Tengo que hacer el forward para cada modelo y ver que clases acierta
            for model_indx, model in enumerate(models):
                if(model.training==True): model.eval()
                if net_type == "fully-connected":
                    model_out = model.forward(Variable(batch.float().view(batch.shape[0], -1).cuda()))
                elif net_type == "convolutional":
                    model_out = model.forward(Variable(batch.float().cuda()))
                else: assert False, "Please define your model type!"

                # Algunos modelos devuelven varias salidas como pueden ser la capa
                # reshape y los logits, etc... Por lo que se establece el standar
                # de que la ultima salida sean los logits del modelo para hacer la clasificacion
                if type(model_out) is list or type(model_out) is tuple:
                    model_out = model_out[-1]

                # Transformamos los logits a salida con el indice con el mayor valor
                #  de las tuplas que continen los logits
                res_topk = np.array(topk_accuracy(model_out, target.cuda(), topk=topk))

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


def evaluate_accuracy_loss_models_generator(models, data, loss, max_data=0, topk=(1,), target_one_hot=False, net_type="convolutional"):
    """Computes the accuracy (sobre 1) over the k top predictions for the specified values of k"""
    # Si paso un modelo y topk(1,5) -> acc1, acc5,
    # Si paso dos modelo y topk(1,5) -> m1_acc1, m1_acc5, m2_acc1, m2_acc5
    with torch.no_grad():

        if type(topk)==int:
            maxk = topk
            topk = (topk,)
        else: maxk = max(topk)

        correct_models, loss_models, total_samples = [0]*len(models), [0]*len(models), 0
        for batch_idx, (batch, target) in enumerate(data):

            if target_one_hot: _, target = target.max(dim=1)
            batch_size = target.size(0)

            # calculo predicciones para el error de test de todos los modelos
            # Tengo que hacer el forward para cada modelo y ver que clases acierta
            for model_indx, model in enumerate(models):
                if(model.training==True): model.eval()
                if net_type == "fully-connected":
                    model_out = model.forward(Variable(batch.float().view(batch.shape[0], -1).cuda()))
                elif net_type == "convolutional":
                    model_out = model.forward(Variable(batch.float().cuda()))
                else: assert False, "Please define your model type!"

                # Algunos modelos devuelven varias salidas como pueden ser la capa
                # reshape y los logits, etc... Por lo que se establece el standar
                # de que la ultima salida sean los logits del modelo para hacer la clasificacion
                if type(model_out) is list or type(model_out) is tuple:
                    model_out = model_out[-1]

                # Transformamos los logits a salida con el indice con el mayor valor
                #  de las tuplas que continen los logits
                res_topk = np.array(topk_accuracy(model_out, target.cuda(), topk=topk))

                correct_models[model_indx] += res_topk

                try: cost = loss(model_out, target.cuda())
                except:
                    global CROSS_ENTROPY_ONE_HOT_WARNING
                    if not CROSS_ENTROPY_ONE_HOT_WARNING:
                        print("\nWARNING-INFO: Crossentropy not works with one hot target encoding!\n")
                        CROSS_ENTROPY_ONE_HOT_WARNING = True
                    cost = loss(model_out, target[:,0])
                loss_models[model_indx] += cost.item()

            total_samples += batch_size
            if max_data != 0 and total_samples >= max_data: break

    """
    accuracies = []
    for result_model in correct_models:
        for topkres in result_model:
            accuracies.append((topkres*1.0)/total_samples)

    #accuracies = list(((np.array(correct_models) * 1.0) / total_samples))
    if len(accuracies) == 1: return accuracies[0]
    return accuracies
    """
    accuracies, losses = [], []
    for indx, result_model in enumerate(correct_models):
        for topkres in result_model:
            accuracies.append((topkres*1.0)/total_samples)
        losses.append((loss_models[indx]*1.0)/total_samples)

    #accuracies = list(((np.array(correct_models) * 1.0) / total_samples))
    if len(accuracies) == 1: return accuracies[0], losses[0]
    return accuracies[0], accuracies[1], losses[0]
    #zipped = [a for a in zip(accuracies,losses)]
    #return [item for sublist in zipped for item in sublist]


def evaluate_accuracy_models_data(models, X_data, y_data, batch_size=100, max_data=0, topk=(1,), net_type="convolutional"):
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
            if total_samples + batch_size >= len(X_data): batch_size = (len(X_data)) - total_samples

            batch = X_data[total_samples:total_samples+batch_size]
            target = y_data[total_samples:total_samples+batch_size]

            # calculo predicciones para el error de test de todos los modelos
            # Tengo que hacer el forward para cada modelo y ver que clases acierta
            for model_indx, model in enumerate(models):
                if(model.training==True): model.eval()
                if net_type == "fully-connected":
                    model_out = model.forward(Variable(batch.float().view(batch.shape[0], -1).cuda()))
                elif net_type == "convolutional":
                    model_out = model.forward(Variable(batch.float().cuda()))
                else: assert False, "Please define your model type!"

                # Algunos modelos devuelven varias salidas como pueden ser la capa
                # reshape y los logits, etc... Por lo que se establece el standar
                # de que la ultima salida sean los logits del modelo para hacer la clasificacion
                if type(model_out) is list or type(model_out) is tuple:
                    model_out = model_out[-1]

                # Transformamos los logits a salida con el indice con el mayor valor
                #  de las tuplas que continen los logits
                res_topk = np.array(topk_accuracy(model_out, target.cuda(), topk=topk))
                correct_models[model_indx] += res_topk


            total_samples+=batch_size
            if max_data != 0 and total_samples >= max_data or total_samples == len(X_data): break

    accuracies = []
    for result_model in correct_models:
        for topkres in result_model:
            accuracies.append((topkres*1.0)/total_samples)

    #accuracies = list(((np.array(correct_models) * 1.0) / total_samples))
    if len(accuracies) == 1: return accuracies[0]
    return accuracies


def evaluate_accuracy_loss_models_data(models, X_data, y_data, loss, batch_size=100, max_data=0, topk=(1,), net_type="convolutional"):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Si paso un modelo y topk(1,5) -> acc1, acc5,
    # Si paso dos modelo y topk(1,5) -> m1_acc1, m1_acc5, m2_acc1, m2_acc5
    with torch.no_grad():

        if type(topk)==int:
            maxk = topk
            topk = (topk,)
        else: maxk = max(topk)

        correct_models, loss_models, total_samples = [0]*len(models), [0]*len(models), 0

        total_samples = 0
        while True:

            # Debemos comprobar que no nos pasamos con el batch_size
            if total_samples + batch_size >= len(X_data): batch_size = (len(X_data)) - total_samples

            batch = X_data[total_samples:total_samples+batch_size]
            target = y_data[total_samples:total_samples+batch_size]

            # calculo predicciones para el error de test de todos los modelos
            # Tengo que hacer el forward para cada modelo y ver que clases acierta
            for model_indx, model in enumerate(models):
                if(model.training==True): model.eval()
                #if(model.training==True): model.eval()

                if net_type == "fully-connected":
                    model_out = model.forward(Variable(batch.float().view(batch.shape[0], -1).cuda()))
                elif net_type == "convolutional":
                    model_out = model.forward(Variable(batch.float().cuda()))
                else: assert False, "Please define your model type!"

                # Algunos modelos devuelven varias salidas como pueden ser la capa
                # reshape y los logits, etc... Por lo que se establece el standar
                # de que la ultima salida sean los logits del modelo para hacer la clasificacion
                if type(model_out) is list or type(model_out) is tuple:
                    model_out = model_out[-1]

                # Transformamos los logits a salida con el indice con el mayor valor
                #  de las tuplas que continen los logits
                res_topk = np.array(topk_accuracy(model_out, target.cuda(), topk=topk))
                correct_models[model_indx] += res_topk

                try: cost = loss(model_out, target.cuda())
                except:
                    global CROSS_ENTROPY_ONE_HOT_WARNING
                    if not CROSS_ENTROPY_ONE_HOT_WARNING:
                        print("\nWARNING-INFO: Crossentropy not works with one hot target encoding!\n")
                        CROSS_ENTROPY_ONE_HOT_WARNING = True
                    cost = loss(model_out, target[:,0])
                loss_models[model_indx] += cost.item()

            total_samples+=batch_size
            if max_data != 0 and total_samples >= max_data or total_samples == len(X_data): break

    accuracies, losses = [], []
    for indx, result_model in enumerate(correct_models):
        for topkres in result_model:
            accuracies.append((topkres*1.0)/total_samples)
        losses.append((loss_models[indx]*1.0)/total_samples)

    #accuracies = list(((np.array(correct_models) * 1.0) / total_samples))
    if len(accuracies) == 1: return accuracies[0], losses[0]
    return accuracies[0], accuracies[1], losses[0]
    #zipped = [a for a in zip(accuracies,losses)]
    #return [item for sublist in zipped for item in sublist]

def evaluate_accuracy_model_predictions(model_out, y_data, batch_size=100, max_data=0, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # Si paso un modelo y topk(1,5) -> acc1, acc5,
    # Solo permite pasar una salida models_out!
    with torch.no_grad():

        if type(topk)==int:
            maxk = topk
            topk = (topk,)
        else: maxk = max(topk)

        # Algunos modelos devuelven varias salidas como pueden ser la capa
        # reshape y los logits, etc... Por lo que se establece el standar
        # de que la ultima salida sean los logits del modelo para hacer la clasificacion
        if type(model_out) is list or type(model_out) is tuple:
            model_out = model_out[-1]

        correct_models, total_samples = 0, 0

        total_samples = 0
        while True:

            # Debemos comprobar que no nos pasamos con el batch_size
            if total_samples + batch_size >= len(model_out): batch_size = (len(model_out)) - total_samples

            batch_out = model_out[total_samples:total_samples+batch_size]
            target = y_data[total_samples:total_samples+batch_size]

            # Transformamos los logits a salida con el indice con el mayor valor
            #  de las tuplas que continen los logits
            res_topk = np.array(topk_accuracy(batch_out, target.cuda(), topk=topk))
            correct_models += res_topk

            total_samples+=batch_size
            if max_data != 0 and total_samples >= max_data or total_samples == len(model_out): break

    return (correct_models*1.0 / total_samples)

def predictions_models_data(models, X_data, batch_size=100, net_type="convolutional"):
    """Computes the predictions for the specified data X_data"""
    with torch.no_grad():

        outs_models, total_samples = [torch.zeros(0,0).cuda()]*len(models), 0

        total_samples = 0
        while True:

            # Debemos comprobar que no nos pasamos con el batch_size
            if total_samples + batch_size >= len(X_data): batch_size = (len(X_data)) - total_samples

            batch = X_data[total_samples:total_samples+batch_size]

            # calculo predicciones para el error de test de todos los modelos
            # Tengo que hacer el forward para cada modelo y ver que clases acierta
            for model_indx, model in enumerate(models):
                if net_type == "fully-connected":
                    model_out = model.forward(Variable(batch.float().view(batch.shape[0], -1).cuda()))
                elif net_type == "convolutional":
                    model_out = model.forward(Variable(batch.float().cuda()))
                else: assert False, "Please define your model type!"

                # Algunos modelos devuelven varias salidas como pueden ser la capa
                # reshape y los logits, etc... Por lo que se establece el standar
                # de que la ultima salida sean los logits del modelo para hacer la clasificacion
                if type(model_out) is list or type(model_out) is tuple:
                    model_out = model_out[-1]

                outs_models[0]=torch.cat((outs_models[0], model_out))

            total_samples+=batch_size
            if total_samples == len(X_data): break

    if len(outs_models) == 1: return outs_models[0]
    return outs_models


# INPUTS: output have shape of [batch_size, category_count]
#    and target in the shape of [batch_size] * there is only one true class for each sample
# topk is tuple of classes to be included in the precision
# topk have to a tuple so if you are giving one number, do not forget the comma
def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    #we do not need gradient calculation for those
    with torch.no_grad():
    #we will use biggest k, and calculate all precisions from 0 to k
        maxk = max(topk)
        batch_size = target.size(0)
        #topk gives biggest maxk values on dimth dimension from output
        #output was [batch_size, category_count], dim=1 so we will select biggest category scores for each batch
        # input=maxk, so we will select maxk number of classes
        #so result will be [batch_size,maxk]
        #topk returns a tuple (values, indexes) of results
        # we only need indexes(pred)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        # then we transpose pred to be in shape of [maxk, batch_size]
        pred = pred.t()
        #we flatten target and then expand target to be like pred
        # target [batch_size] becomes [1,batch_size]
        # target [1,batch_size] expands to be [maxk, batch_size] by repeating same correct class answer maxk times.
        # when you compare pred (indexes) with expanded target, you get 'correct' matrix in the shape of  [maxk, batch_size] filled with 1 and 0 for correct and wrong class assignments
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        """ correct=([[0, 0, 1,  ..., 0, 0, 0],
        [1, 0, 0,  ..., 0, 0, 0],
        [0, 0, 0,  ..., 1, 0, 0],
        [0, 0, 0,  ..., 0, 0, 0],
        [0, 1, 0,  ..., 0, 0, 0]], device='cuda:0', dtype=torch.uint8) """
        res = []
        # then we look for each k summing 1s in the correct matrix for first k element.
        for k in topk:
            res.append(correct[:k].view(-1).float().sum(0, keepdim=True))
        return res


##########################################################################################################
##########################################################################################################
##########################################################################################################

def findLR(model, optimizer, criterion, trainloader, final_value=10, init_value=1e-8, verbose=1):
    #https://medium.com/coinmonks/training-neural-networks-upto-10x-faster-3246d84caacd
    '''
      findLR plots the graph for the optimum learning rates for the model with the
      corresponding dataset.
      The technique is quite simple. For one epoch,
      1. Start with a very small learning rate (around 1e-8) and increase the learning rate linearly.
      2. Plot the loss at each step of LR.
      3. Stop the learning rate finder when loss stops going down and starts increasing.

      A graph is created with the x axis having learning rates and the y axis
      having the losses.

      Arguments:
      1. model -  (torch.nn.Module) The deep learning pytorch network.
      2. optimizer: (torch.optim) The optimiser for the model eg: SGD,CrossEntropy etc
      3. criterion: (torch.nn) The loss function that is used for the model.
      4. trainloader: (torch.utils.data.DataLoader) The data loader that loads data in batches for input into model
      5. final_value: (float) Final value of learning rate
      6. init_value: (float) Starting learning rate.

      Returns:
       learning rates used and corresponding losses


    '''
    model.train() # setup model for training configuration

    num = len(trainloader) - 1 # total number of batches
    mult = (final_value / init_value) ** (1/num)

    losses = []
    lrs = []
    best_loss = 0.
    avg_loss = 0.
    beta = 0.98 # the value for smooth losses
    lr = init_value

    for batch_num, (inputs, targets) in enumerate(trainloader):

        if verbose==1: print("Testint LR: {}".format(lr))
        optimizer.param_groups[0]['lr'] = lr
        batch_num += 1 # for non zero value
        inputs, targets = inputs.cuda(), targets.cuda() # convert to cuda for GPU usage
        optimizer.zero_grad() # clear gradients
        outputs = model(inputs) # forward pass
        loss = criterion(outputs, targets.long().cuda()) # compute loss

        #Compute the smoothed loss to create a clean graph
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)

        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss

        # append loss and learning rates for plotting
        lrs.append(math.log10(lr))
        losses.append(smoothed_loss)

        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        # backprop for next step
        loss.backward()
        optimizer.step()

        # update learning rate
        lr = mult*lr

    #plt.xlabel('Learning Rates')
    #plt.ylabel('Losses')
    #plt.plot(lrs,losses)
    #plt.show()
    return lrs, losses

##########################################################################################################
##########################################################################################################
##########################################################################################################

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


'''
mixup: BEYOND EMPIRICAL RISK MINIMIZATION: https://arxiv.org/abs/1710.09412
https://github.com/facebookresearch/mixup-cifar10
'''

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

### Ejemplo de uso
# inputs, targets_a, targets_b, lam = mixup_data(batch_data, batch_target, alpha_mixup)
# inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))

# outputs = model(inputs)
# loss = mixup_criterion(loss_ce, outputs, targets_a, targets_b, lam)
# total_loss += loss.item()


''' ######################################################################## '''
''' #############################  CUTOUT ################################## '''
''' ######################################################################## '''
# https://github.com/uoguelph-mlrg/Cutout
# Para usarlo si estamos usando albumentations añadir otro transform separado que sea
# por ejemplo transforms_torchvision y a traves de ese lo usamos como self.torchvision_transform(feature)
# Hay un ejemplo en el dataloader de LFW -> data_generator.py -> NPDatasetLFW
class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img