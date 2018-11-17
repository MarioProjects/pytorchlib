import os
import cv2
import pandas as pd
import json
import numpy as np

import torch
from torch.utils import data

import pytorchlib.pytorch_library.utils_particular as utils_particular
import pytorchlib.pytorch_library.utils_training as utils_training

import albumentations

import PIL

def load_img(path):
    return np.array(PIL.Image.open(path).convert('RGB'))

class FoldersDataset(data.Dataset):
    """
        Cargador de prueba de dataset almacenado en carpetas del modo
            -> train/clase1/TodasImagenes ...
        data_path: ruta a la carpeta padre del dataset. Ejemplo train/
        transforms: lista de transformaciones de albumentations a aplicar
        cat2class: diccionario con claves clas clases y valor la codificacion
            de cada clase. Ejemplo {'perro':0, 'gato':1}
    """
    def __init__(self, data_path, transforms=[], cat2class=[], normalization=""):
        different_classes, all_paths, all_classes = [], [], []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                fullpath = os.path.join(path, name)
                current_class = fullpath.split("/")[-2]
                all_paths.append(fullpath)
                all_classes.append(current_class)
                if current_class not in different_classes: different_classes.append(current_class)

        #class2cat = dict(zip(np.arange(0, len(different_classes)), different_classes))
        if cat2class==[]: cat2class = dict(zip(different_classes, np.arange(0, len(different_classes))))

        for indx, c_class in enumerate(all_classes):
            all_classes[indx] = cat2class[c_class]
        
        self.imgs_paths = all_paths
        self.labels = all_classes
        self.transforms = transforms
        self.norm= normalization
        
    def __getitem__(self,index):
        img = load_img(self.imgs_paths[index])
        
        if self.transforms!=[]:
            for transform in self.transforms:
                img = apply_img_albumentation(transform, img)
                
        img = torch.from_numpy(img.astype(np.float32).transpose(2,0,1))
        if self.norm!="": img = single_normalize(img, self.norm)
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs_paths)


def dataloader_from_numpy(features, targets, batch_size, transforms=[], seed=0, norm="", num_classes=0):
    """ WARNING: ESTO DEBERIA PODER MEJORAR Y HACER UN DATASET CORRECTAMENTE (mirar FoldersDataset)
    Generador de loaders generico para numpy arrays
    transforms: lista con transformaciones albumentations (LISTA y no Compose!)
    """
    generator = np.random.RandomState(seed=seed)
    pick_order = n_samples = len(features)
    pick_per_epoch = n_samples // batch_size

    #print("""WARNING: This function (dataloader_from_numpy) receives the features with the form [batch, width, height, channels]
    #        and internally transposes these features to [batch, channels, width, height]""")

    while True:  # Infinity loop
        pick_order = generator.permutation(pick_order)
        for i in range(pick_per_epoch):
            current_picks = pick_order[i*batch_size: (i+1)*batch_size]
            current_features = features[current_picks]
            current_targets = targets[current_picks]

            # Debemos aplicar las transformaciones pertinentes definidas en transforms (albumentations)
            current_features_transformed = []
            if transforms!=[]:
                for indx, (sample) in enumerate(current_features):
                    for transform in transforms:
                        sample = apply_img_albumentation(transform, sample)
                    current_features_transformed.append(sample)

            # Para evitar problemas con imagenes en blanco y negro (1 canal)
            if current_features_transformed!=[]: current_features = np.array(current_features_transformed)
            if len(current_features.shape) == 3: current_features = np.expand_dims(current_features, axis=3)

            current_features = torch.from_numpy(current_features)
            current_targets = utils_training.to_categorical(current_targets, num_classes=num_classes)
            current_targets = torch.from_numpy(current_targets)

            current_features = current_features.permute(0,3,1,2)

            # Normalizamos los datos
            if norm != "":
                current_features = single_normalize(current_features, norm)
            yield current_features, current_targets


def normalize(tr_feat, te_feat, norm):
    if norm=='zscore':
        mean=torch.ones(1,3,1,1)
        std=torch.ones(1,3,1,1)
        mean[0,0,0,0]=torch.mean(tr_feat[:,0,:,:])
        mean[0,1,0,0]=torch.mean(tr_feat[:,1,:,:])
        mean[0,2,0,0]=torch.mean(tr_feat[:,2,:,:])
        std[0,0,0,0]=torch.std(tr_feat[:,0,:,:])
        std[0,1,0,0]=torch.std(tr_feat[:,1,:,:])
        std[0,2,0,0]=torch.std(tr_feat[:,2,:,:])
        tr_feat-=mean
        tr_feat/=std
        te_feat-=mean
        te_feat/=std

    elif norm=='0_1range':
        max_val=tr_feat.max()
        min_val=tr_feat.min()
        tr_feat-=min_val
        tr_feat/=(max_val-min_val)

        max_val=te_feat.max()
        min_val=te_feat.min()
        te_feat-=min_val
        te_feat/=(max_val-min_val)


    elif norm=='-1_1range' or norm=='np_range':
        max_val=tr_feat.max()
        min_val=tr_feat.min()
        tr_feat*=2
        tr_feat-=(max_val-min_val)
        tr_feat/=(max_val-min_val)

        max_val=te_feat.max()
        min_val=te_feat.min()
        te_feat*=2
        te_feat-=(max_val-min_val)
        te_feat/=(max_val-min_val)

    elif norm=='255':
        tr_feats/=255
        te_feats/=255

    else: assert False, "Invalid Normalization"

    return tr_feat, te_feat

def single_normalize(feats, norm):

    if norm=='0_1range':
        max_val=feats.max()
        min_val=feats.min()
        feats-=min_val
        feats/=(max_val-min_val)


    elif norm=='-1_1range' or norm=='np_range':
        max_val=feats.max()
        min_val=feats.min()
        feats*=2
        feats-=(max_val-min_val)
        feats/=(max_val-min_val)

    elif norm=='255':
        feats/=255

    elif norm==None: pass
    else: raise NotImplemented

    return feats


def apply_img_albumentation(aug, image):
    image = aug(image=image)['image']
    return image