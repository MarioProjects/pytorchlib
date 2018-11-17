import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from slackclient import SlackClient
from torch.autograd.variable import Variable

def slack_message(message, channel):
    token = 'xoxp-458177397862-456198240913-464006277364-a7557a18c11f5e99ca9ce33deacbefc4'
    sc = SlackClient(token)
    sc.api_call('chat.postMessage', channel=channel,
                text=message, username='My Sweet Bot',
                icon_emoji=':robot_face:')

def images_to_vectors(images):
    """
    Con esta funcion pasamos las imagenes a vectores -> Flatten. Tensores.
    """
    return images.view(images.size(0), -1).type(torch.cuda.FloatTensor)


def vectors_to_images(vectors, width, height, depth):
    """
    Con esta funcion pasamos los vectores a imagenes
    """
    return vectors.view(vectors.size(0), depth, width, height).type(torch.cuda.FloatTensor)

def normal_noise(samples, out_features):
    """
    Con esta funcion creamos tantas muestras como 'samples'
    formadas por ruido gausiano con una distribucion normal
    con media 0 y varianza 1. Dimensionalidad out_features 
    (la misma que toma el generador para samplear muestras)
    """
    return Variable(torch.randn(samples, out_features)).type(torch.cuda.FloatTensor)


def time_to_human(start,end):
    #### start and end --> time.time()
    #### returns string
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    #print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    #print("{:0>1} hr {:0>2} min".format(int(hours),int(minutes)))
    return "{:0>1} hr {:0>2} min".format(int(hours),int(minutes))