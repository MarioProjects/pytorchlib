import os
import cv2
import pandas as pd
import json
import numpy as np

import torch
from torch.utils import data

import pytorchlib.pytorch_data.transforms as custom_transforms
import pytorchlib.pytorch_library.utils_particular as utils_particular
import pytorchlib.pytorch_library.utils_training as utils_training


import PIL

def load_img(path):
    return np.array(PIL.Image.open(path).convert('RGB'))

class FoldersDatasetVGGFace2(data.Dataset):
    """
        Cargador de prueba de dataset almacenado en carpetas del modo
            -> train/clase1/TodasImagenes ...
        data_path: ruta a la carpeta padre del dataset. Ejemplo train/
        transforms: lista de transformaciones de albumentations a aplicar
        cat2class: diccionario con claves clas clases y valor la codificacion
            de cada clase. Ejemplo {'perro':0, 'gato':1}
    """
    def __init__(self, data_path, transforms=[], cat2class=[]):
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
        
        """ https://arxiv.org/pdf/1710.08092.pdf
        Training implementation details. All the networks are
        trained for classification using the soft-max loss function.
        During training, the extended bounding box of the face is
        resized so that the shorter side is 256 pixels, then a 224×224
        pixels region is randomly cropped from each sample. The
        mean value of each channel is subtracted for each pixel.
        Monochrome augmentation is used with a probability of
        20% to reduce the over-fitting on colour images. Stochastic
        gradient descent is used with mini-batches of size 256, with
        a balancing-sampling strategy for each mini-batch due to the
        unbalanced training distributions. The initial learning rate is
        0.1 for the models trained from scratch, and this is decreased
        twice with a factor of 10 when errors plateau. The weights
        of the models are initialised as described in [8]. The learning
        rate for model fine-tuning starts from 0.005 and decreases
        to 0.001
        """
        self.transforms = transforms
        
        
    def __getitem__(self,index):
        img = load_img(self.imgs_paths[index])
        
        if self.transforms!=[]:
            for transform in self.transforms:
                img = custom_transforms.apply_albumentation(transform, img)
                
        return torch.from_numpy(img).permute(2,0,1), self.labels[index]

    def __len__(self):
        return len(self.imgs_paths)