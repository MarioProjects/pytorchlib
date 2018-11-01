import os
import gzip
import time
import numpy as np
import pandas as pd
import pickle

# https://albumentations.readthedocs.io/en/latest/api/index.html
import albumentations

import torch
import torch.utils.data as torchdata
from torch import nn, optim

from pytorchlib.pytorch_data.dataloader import ConvolutionalDataset, gen_quick_draw_doodle
import pytorchlib.pytorch_data.transforms as custom_transforms

import scipy.io
import scipy.misc
from sklearn.model_selection import train_test_split

#########################
# DATSETS PARA SAMPLEAR
##########################
# a convolutional data wrapper to apply torchvision transformations to home made datasets

def lfw_gender(gray, seed=0, fold_test=0, batch_size=128, norm="None"):
    '''
    LOAD LFW DATASET
    Carga los 5 folds de LFW y y de ellos toma 1 ("fold_test") como test
    '''
    if gray: data_path = '/home/maparla/DeepLearning/PytorchDatasets/nplfwdeepfunneled_gray/'
    else: data_path = '/home/maparla/DeepLearning/PytorchDatasets/nplfwdeepfunneled/'

    """ Extraemos los datos """
    te_feat, tr_feat = [], []
    te_labl, tr_labl = [], []

    if gray: load_gray = "_gray"
    else: load_gray = ""

    for i in range(0,5):
        if int(fold_test)==i:
            te_feat = np.load(data_path+"fold"+str(i)+"_data"+load_gray+".npy")
            te_labl = np.load(data_path+"fold"+str(i)+"_labels.npy")
        else:
            tr_feat.append(np.load(data_path+"fold"+str(i)+"_data"+load_gray+".npy"))
            tr_labl.append(np.load(data_path+"fold"+str(i)+"_labels.npy"))

    tr_feat = np.concatenate(tr_feat)
    tr_labl = np.concatenate(tr_labl)

    # tr_feat = tr_feat.astype('float32').transpose(0,3,1,2)
    # Debemos hace el transpose para poner los canales delante
    tr_feat = tr_feat.astype('float32').transpose(0,3,1,2)
    tr_labl = tr_labl.astype('int64').reshape(-1, 1)
    te_feat = te_feat.astype('float32').transpose(0,3,1,2)
    te_labl = te_labl.astype('int64').reshape(-1, 1)

    """ Ordenamos los datos según la semilla, los barajamos (Train) """
    generator = np.random.RandomState(seed=seed)
    index = generator.permutation(tr_feat.shape[0])
    tr_feat = tr_feat[index]
    tr_labl = tr_labl[index]

    """ Ordenamos los datos según la semilla, los barajamos (Test) """
    generator = np.random.RandomState(seed=seed)
    index = generator.permutation(te_feat.shape[0])
    te_feat = te_feat[index]
    te_labl = te_labl[index]

    # Los pasamos a torch los datos
    tr_feat = torch.from_numpy(tr_feat).cuda()
    tr_labl = torch.from_numpy(tr_labl).cuda()
    te_feat = torch.from_numpy(te_feat).cuda()
    te_labl = torch.from_numpy(te_labl).cuda()
    # Normalizamos los datos
    tr_feat, te_feat = custom_transforms.normalize(tr_feat, te_feat, norm)

    """ DATALOADER DEL TRAINING """
    train_trans=[custom_transforms.RandomHorizontalFlip(100), custom_transforms.RandomRotation(7,seed=seed), 
                custom_transforms.RandomCrop(80,seed=seed), custom_transforms.ToGPU()]
    transforms_train = custom_transforms.Compose(train_trans)
    train_set = ConvolutionalDataset(tr_feat,tr_labl,transforms=transforms_train,tag_transforms=None)
    train_loader=torchdata.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)

    """ DATALOADER DE TEST """
    test_trans=[custom_transforms.CentralCrop(80), custom_transforms.ToGPU()]
    transforms_eval = custom_transforms.Compose(test_trans)
    test_set = ConvolutionalDataset(te_feat, te_labl,transforms=transforms_eval,tag_transforms=None)
    test_loader=torchdata.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)
    train_samples = len(tr_feat)
    test_samples = len(te_feat)

    return train_samples, train_loader, test_samples, test_loader

def lfw_gender_test(gray, seed=0, fold_test=0, batch_size=128, norm="None"):
    '''
    LOAD LFW DATASET
    Carga los folds de LFW y y de ellos toma 1 ("fold_test") como test
    '''
    if gray: data_path = '/home/maparla/DeepLearning/PytorchDatasets/nplfwdeepfunneled_gray/'
    else: data_path = '/home/maparla/DeepLearning/PytorchDatasets/nplfwdeepfunneled/'

    """ Extraemos los datos """
    te_feat, te_labl = [], []

    if gray: load_gray = "_gray"
    else: load_gray = ""

    for i in range(0,5):
        if int(fold_test)==i:
            te_feat = np.load(data_path+"fold"+str(i)+"_data"+load_gray+".npy")
            te_labl = np.load(data_path+"fold"+str(i)+"_labels.npy")
        else: pass

    # Debemos hace el transpose para poner los canales delante
    te_feat = te_feat.astype('float32').transpose(0,3,1,2)
    te_labl = te_labl.astype('int64').reshape(-1, 1)

    """ Ordenamos los datos según la semilla, los barajamos (Test) """
    generator = np.random.RandomState(seed=seed)
    index = generator.permutation(te_feat.shape[0])
    te_feat = te_feat[index]
    te_labl = te_labl[index]

    # Los pasamos a torch los datos
    te_feat = torch.from_numpy(te_feat).cuda()
    te_labl = torch.from_numpy(te_labl).cuda()
    # Normalizamos los datos
    te_feat = custom_transforms.single_normalize(te_feat, norm)


    """ DATALOADER DE TEST """
    test_trans=[custom_transforms.CentralCrop(80), custom_transforms.ToGPU()]
    transforms_eval = custom_transforms.Compose(test_trans)
    test_set = ConvolutionalDataset(te_feat, te_labl,transforms=transforms_eval,tag_transforms=None)
    test_loader=torchdata.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

    test_samples = len(te_feat)

    return test_samples, test_loader

def groups_gender(gray, seed=0, fold_test=0, batch_size=128, norm="None"):
    '''
    LOAD GROUPS DATASET
    Carga los 5 folds de GROUPS y y de ellos toma 1 ("fold_test") como test
    '''

    if gray: data_path = '/home/maparla/DeepLearning/PytorchDatasets/npgroups_gray/'
    else: data_path = '/home/maparla/DeepLearning/PytorchDatasets/npgroups/'

    """ Extraemos los datos """
    te_feat, tr_feat = [], []
    te_labl, tr_labl = [], []

    if gray: load_gray = "_gray"
    else: load_gray = ""

    for i in range(0,5):
        if int(fold_test)==i:
            te_feat = np.load(data_path+"fold"+str(i)+"_data"+load_gray+".npy")
            te_labl = np.load(data_path+"fold"+str(i)+"_labels.npy")
        else:
            tr_feat.append(np.load(data_path+"fold"+str(i)+"_data"+load_gray+".npy"))
            tr_labl.append(np.load(data_path+"fold"+str(i)+"_labels.npy"))

    tr_feat = np.concatenate(tr_feat)
    tr_labl = np.concatenate(tr_labl)

    # tr_feat = tr_feat.astype('float32').transpose(0,3,1,2)
    # Debemos hace el transpose para poner los canales delante
    tr_feat = tr_feat.astype('float32').transpose(0,3,1,2)
    tr_labl = tr_labl.astype('int64').reshape(-1, 1)
    te_feat = te_feat.astype('float32').transpose(0,3,1,2)
    te_labl = te_labl.astype('int64').reshape(-1, 1)

    """ Ordenamos los datos según la semilla, los barajamos (Train) """
    generator = np.random.RandomState(seed=seed)
    index = generator.permutation(tr_feat.shape[0])
    tr_feat = tr_feat[index]
    tr_labl = tr_labl[index]

    """ Ordenamos los datos según la semilla, los barajamos (Test) """
    generator = np.random.RandomState(seed=seed)
    index = generator.permutation(te_feat.shape[0])
    te_feat = te_feat[index]
    te_labl = te_labl[index]

    # Los pasamos a torch los datos
    tr_feat = torch.from_numpy(tr_feat).cuda()
    tr_labl = torch.from_numpy(tr_labl).cuda()
    te_feat = torch.from_numpy(te_feat).cuda()
    te_labl = torch.from_numpy(te_labl).cuda()
    # Normalizamos los datos
    tr_feat, te_feat = custom_transforms.normalize(tr_feat, te_feat, norm)

    """ DATALOADER DEL TRAINING """
    train_trans=[custom_transforms.RandomHorizontalFlip(100), custom_transforms.RandomRotation(7,seed=seed), custom_transforms.RandomCrop(80,seed=seed), custom_transforms.ToGPU()]
    transforms_train = custom_transforms.Compose(train_trans)
    train_set = ConvolutionalDataset(tr_feat,tr_labl,transforms=transforms_train,tag_transforms=None)
    train_loader=torchdata.DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)

    """ DATALOADER DE TEST """
    test_trans=[custom_transforms.CentralCrop(80), custom_transforms.ToGPU()]
    transforms_eval = custom_transforms.Compose(test_trans)
    test_set = ConvolutionalDataset(te_feat, te_labl,transforms=transforms_eval,tag_transforms=None)
    test_loader=torchdata.DataLoader(dataset=test_set,batch_size=batch_size,shuffle=False)

    train_samples = len(tr_feat)
    test_samples = len(te_feat)

    return train_samples, train_loader, test_samples, test_loader


def quick_draw_doodle(seed=0, train_segment="0_10000", batch_size=100, norm="None", desired_img_size=(224, 224), evaluation_mode=False):
    '''
    LOAD Quick Draw Doodle DATASET
    Carga los dataos de Quick Draw Doodle - https://www.kaggle.com/c/quickdraw-doodle-recognition
    '''

    if evaluation_mode:
        data_type = "eval"
        probability_apply_transformations = 1
        transforms = albumentations.Compose([
            #albumentations.CenterCrop(desired_img_size[1], desired_img_size[0]),
            albumentations.Normalize(max_pixel_value = 255),
        ], p=probability_apply_transformations)
    else:
        data_type = "train"
        probability_apply_transformations = 1
        transforms = albumentations.Compose([
            #albumentations.RandomCrop(desired_img_size[1], desired_img_size[0]),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.Normalize(max_pixel_value = 255),
        ], p=probability_apply_transformations)


    data_path = '/home/maparla/DeepLearning/KaggleDatasets/quick_draw_doodle/train_simplified_groups/'
    df = pd.read_csv(data_path + data_type + '_'+train_segment+'.csv', sep="\t", encoding='utf-8')

    generator = np.random.RandomState(seed=seed)

    n_samples = df.shape[0]

    pick_order = np.arange(n_samples)
    pick_per_epoch = n_samples // batch_size

    dataloaders = gen_quick_draw_doodle(df, pick_order, pick_per_epoch, batch_size, generator, transforms, norm)
    return len(df), dataloaders
