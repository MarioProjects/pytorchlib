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

CAT2CLASS = {'affenpinscher': 22,'afghan_hound': 67,'african_hunting_dog': 10,'airedale': 87,'american_staffordshire_terrier': 45,'appenzeller': 90,'australian_terrier': 49,'basenji': 111,'basset': 86,'beagle': 89,'bedlington_terrier': 116,'bernese_mountain_dog': 34,'black-and-tan_coonhound': 27,'blenheim_spaniel': 28,'bloodhound': 52,'bluetick': 108,'border_collie': 48,'border_terrier': 109,'borzoi': 57,'boston_bull': 4,'bouvier_des_flandres': 36,'boxer': 20,'brabancon_griffon': 35,'briard': 83,'brittany_spaniel': 41,'bull_mastiff': 30,'cairn': 50,'cardigan': 94,'chesapeake_bay_retriever': 106,'chihuahua': 39,'chow': 5,'clumber': 29,'cocker_spaniel': 117,'collie': 75,'curly-coated_retriever': 85,'dandie_dinmont': 82,'dhole': 65,'dingo': 33,'doberman': 91,'english_foxhound': 68,'english_setter': 76,'english_springer': 80,'entlebucher': 51,'eskimo_dog': 62,'flat-coated_retriever': 8,'french_bulldog': 56,'german_shepherd': 84,'german_short-haired_pointer': 3,'giant_schnauzer': 21,'golden_retriever': 7,'gordon_setter': 44,'great_dane': 54,'great_pyrenees': 16,'greater_swiss_mountain_dog': 71,'groenendael': 69,'ibizan_hound': 79,'irish_setter': 42,'irish_terrier': 53,'irish_water_spaniel': 40,'irish_wolfhound': 100,'italian_greyhound': 88,'japanese_spaniel': 31,'keeshond': 19,'kelpie': 55,'kerry_blue_terrier': 47,'komondor': 59,'kuvasz': 25,'labrador_retriever': 81,'lakeland_terrier': 103,'leonberg': 70,'lhasa': 92,'malamute': 60,'malinois': 105,'maltese_dog': 11,'mexican_hairless': 58,'miniature_pinscher': 96,'miniature_poodle': 14,'miniature_schnauzer': 112,'newfoundland': 37,'norfolk_terrier': 1,'norwegian_elkhound': 13,'norwich_terrier': 115,'old_english_sheepdog': 43,'otterhound': 73,'papillon': 46,'pekinese': 6,'pembroke': 64,'pomeranian': 32,'pug': 113,'redbone': 101,'rhodesian_ridgeback': 63,'rottweiler': 110,'saint_bernard': 61,'saluki': 98,'samoyed': 72,'schipperke': 74,'scotch_terrier': 95,'scottish_deerhound': 9,'sealyham_terrier': 17,'shetland_sheepdog': 23,'shih-tzu': 12,'siberian_husky': 118,'silky_terrier': 97,'soft-coated_wheaten_terrier': 24,'staffordshire_bullterrier': 102,'standard_poodle': 78,'standard_schnauzer': 18,'sussex_spaniel': 2,'tibetan_mastiff': 26,'tibetan_terrier': 99,'toy_poodle': 114,'toy_terrier': 77,'vizsla': 119,'walker_hound': 104,'weimaraner': 93,'welsh_springer_spaniel': 38,'west_highland_white_terrier': 15,'whippet': 107,'wire-haired_fox_terrier': 0,'yorkshire_terrier': 66}

IMG_BASE_SIZE = 100

def load_img(path):
    return np.array(PIL.Image.open(path).convert('RGB'))

class FoldersDatasetDogBreed(data.Dataset):
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
        #cat2class = dict(zip(different_classes, np.arange(0, len(different_classes))))
        if cat2class==[]: cat2class = CAT2CLASS

        for indx, c_class in enumerate(all_classes):
            all_classes[indx] = cat2class[c_class]

        self.imgs_paths = all_paths
        self.labels = all_classes

        self.norm = normalization
        self.transforms = transforms


    def __getitem__(self,index):
        img = load_img(self.imgs_paths[index])

        """ 
        Training implementation details. IMG is
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

        # Primero debemos redimensionar la imagen para que el lado corto mida IMG_BASE_SIZE pixels
        if img.shape[0] <= img.shape[1]: # Tiene menor o igual el alto
            ancho = int((IMG_BASE_SIZE * img.shape[1]) / img.shape[0])
            img = load_data.apply_img_albumentation(albumentations.Resize(IMG_BASE_SIZE, ancho), img)
        else:
            alto = int((IMG_BASE_SIZE * img.shape[0]) / img.shape[1])
            img = load_data.apply_img_albumentation(albumentations.Resize(alto, IMG_BASE_SIZE), img)

        if self.transforms!=[]:
            for transform in self.transforms:
                img = load_data.apply_img_albumentation(transform, img)

        img = torch.from_numpy(img.astype(np.float32).transpose(2,0,1))
        if self.norm!="": img = load_data.single_normalize(img, self.norm)
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs_paths)