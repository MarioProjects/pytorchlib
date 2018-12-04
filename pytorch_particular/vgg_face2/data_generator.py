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

CAT2CLASS = {"male": 0, "female": 1}
CLASS2CAT = {0 : "male", 1 : "female"}

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
    def __init__(self, data_path, bb_path="", transforms=[], cat2class=[], normalization="", img_base_size=100):
        different_classes, all_paths, all_classes = [], [], []
        for path, subdirs, files in os.walk(data_path):
            for name in files:
                fullpath = os.path.join(path, name)
                current_class = fullpath.split("/")[-2]
                all_paths.append(fullpath)
                all_classes.append(current_class)
                if current_class not in different_classes: different_classes.append(current_class)

        #class2cat = dict(zip(np.arange(0, len(different_classes)), different_classes))
        if cat2class==[]: cat2class = CAT2CLASS

        for indx, c_class in enumerate(all_classes):
            all_classes[indx] = cat2class[c_class]

        self.imgs_paths = all_paths
        self.labels = all_classes

        self.norm = normalization
        self.transforms = transforms
        self.img_base_size = img_base_size
        if bb_path != "": self.bounding_boxes = pd.read_csv(bb_path, index_col="NAME_ID")
        else: self.bounding_boxes = ""


    def __getitem__(self,index):
        img_path = self.imgs_paths[index]
        img_id = img_path.split("/")[-1][0:-4]
        img = load_img(img_path)

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

        if type(self.bounding_boxes) != type(""):
            # INFORMACION DE LOS BOUNDING BOXES
            img_info_bb = self.bounding_boxes.loc[img_id]
            sample_x = img_info_bb["X"]
            extend_x = 0 if img_info_bb["X"] >=0 else -img_info_bb["X"]

            sample_y = img_info_bb["Y"]
            extend_y = 0 if img_info_bb["Y"] >=0 else -img_info_bb["Y"]

            sample_w = img_info_bb["W"]
            sample_h = img_info_bb["H"]

            BLACK = [0,0,0]
            if extend_x!=0 or extend_y!=0:
                img = cv2.copyMakeBorder(img, int(extend_x/2), int(extend_x/2), int(extend_y/2), int(extend_y/2),
                                                cv2.BORDER_CONSTANT, value=BLACK)
                if extend_x!=0: sample_x = int(extend_x/2)
                if extend_y!=0: sample_y = int(extend_y/2)
            img = img[sample_y:sample_y+sample_h-extend_y, sample_x:sample_x+sample_w-extend_x]

        # Debemos redimensionar la imagen para que el lado corto mida IMG_BASE_SIZE pixels
        if img.shape[0] <= img.shape[1]: # Tiene menor o igual el alto
            ancho = int((self.img_base_size * img.shape[1]) / img.shape[0])
            img = load_data.apply_img_albumentation(albumentations.Resize(self.img_base_size, ancho), img)
        else:
            alto = int((self.img_base_size * img.shape[0]) / img.shape[1])
            img = load_data.apply_img_albumentation(albumentations.Resize(alto, self.img_base_size), img)
        if self.transforms!=[]:
            for transform in self.transforms:
                img = load_data.apply_img_albumentation(transform, img)

        img = torch.from_numpy(img.astype(np.float32).transpose(2,0,1))
        if self.norm!="": img = load_data.single_normalize(img, self.norm)
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs_paths)