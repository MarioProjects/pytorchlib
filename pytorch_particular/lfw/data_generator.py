import os
import cv2
import pandas as pd
import json
import numpy as np

import torch
from torch.utils import data

import pytorchlib.pytorch_data.load_data as load_data
import pytorchlib.pytorch_library.utils_training as utils_training

import albumentations

import PIL

IMG_BASE_SIZE = 100

class NPDatasetLFW(data.Dataset):
    """
        Cargador de prueba de dataset almacenado en carpetas del modo
            -> train/clase1/TodasImagenes ...
        data_path: ruta a la carpeta padre del dataset. Ejemplo train/
        transforms: lista de transformaciones de albumentations a aplicar
        cat2class: diccionario con claves clas clases y valor la codificacion
            de cada clase. Ejemplo {'perro':0, 'gato':1}
    """
    def __init__(self, data_path, transforms=[], cat2class=[], normalization="", seed=0):
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