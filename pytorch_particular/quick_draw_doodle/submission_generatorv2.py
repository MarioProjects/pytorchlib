#!/usr/bin/env python
# coding: utf-8


import os
import cv2
import pandas as pd
import json
import numpy as np
import pickle

import matplotlib.pyplot as plt
import torch

import pytorchlib.pytorch_library.utils_training as utils_training
from pytorchlib.pytorch_models import models_interface

from pytorchlib.pytorch_particular.quick_draw_doodle import data_generator as data_doodle
from pytorchlib.pytorch_particular.quick_draw_doodle import utils as utils_doodle



import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Model train for Quick Draw Doodle Competition by Mario Parren~o. Msc student Universidad Politecnica de Valencia maparla@inf.upv.es . Enjoy!')
    parser.add_argument('--optimizer', type=str, choices=['SGD','Adam','RMSprop'], default="SGD", help='optimizer to apply')
    parser.add_argument('--model_type', type=str, required=True, default="", help='model type to use')
    parser.add_argument('--model_cfg', type=str, required=True, default="", help='model configuration to use')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch Size for training')
    parser.add_argument('--imgs_size', type=int, required=True, help='Size of images to work with')
    parser.add_argument('--flat_size', type=int, required=False, help='Input features to Fully-Connected part after convolutions')

    parser.add_argument('--block_type', type=str, default="", required=False, help='Block type for ResNet and SENet model')
    parser.add_argument('--last_pool_size', type=int, default=0, required=False, help='Last pool size of MobileNetv2 net')
    #parser.add_argument('--norm', type=str, choices=["zscore","0_1range","-1_1range", "np_range"], default="0_1range", help='Normalization to apply to the data')
    #parser.add_argument('--growth_rate', type=int, default=0, help='Growth rate for DenseNet models')
    aux=parser.parse_args()
    arguments=list()
    arguments=[aux.model_type, aux.model_cfg, aux.optimizer, aux.batch_size, aux.imgs_size, aux.flat_size, aux.block_type, aux.last_pool_size]
    return arguments

model_type, model_cfg, optimizador, batch_size, size_imgs, flat_size, block_type, last_pool_size = parse_args()


test_path = "/home/maparla/DeepLearning/KaggleDatasets/quick_draw_doodle/test_simplified.csv"


test_transforms = []
test_df = pd.read_csv(test_path)


#model_type = "SENet"
#block_type = "PreActBlock"
model_cfg = "50_Standard"
#flat_size = 2048
#last_pool_size = 8
num_classes = 340
#optimizador = "SGD"
da = "" # Data augmentation "" or No DA -> "_noDA"


#size_imgs = 64
norm = "255"
color = False
trazos = [3]


if "_" in model_cfg: model_config = model_cfg.split("_")
else: model_config = model_cfg


log_path = "results/CE_Simple/"+optimizador+"/"+model_type+"/IMG"+str(size_imgs)+"/Trazos/"+block_type+str(model_cfg)+da+"/CE_Simple_LOG.pkl"

with open(log_path, 'rb') as f:
    log_info = pickle.load(f)


print("Maximo valor acc1: {:.2f}%".format(np.array(log_info['log-acc1']).max()*100))
print("Maximo valor acc3: {:.2f}%".format(np.array(log_info['log-acc3']).max()*100))

states_path = "results/CE_Simple/"+optimizador+"/"+model_type+"/IMG"+str(size_imgs)+"/Trazos/"+block_type+str(model_cfg)+da+"/CE3_Simple_checkpoint_state_23-11-2018_08y23.pt"



model_path = 'results/CE_Simple_old/SGD/ResNet/IMG64/BasicBlock34_Small/Retrain/CE_Simple_checkpoint_model.pt'



model = models_interface.load_model(model_type, model_config=model_config, last_pool_size=last_pool_size,
                                    states_path=states_path, flat_size=flat_size,
                                    block_type=block_type, input_channels=4, out_features=num_classes).cuda()


x_test = data_doodle.df_to_image_array_doodle(test_df.loc[0:(20000-1), :], size_imgs, 
                                                    transforms=test_transforms, norm=norm, 
                                                    color=color, trazos=trazos)
test_predictions_first = utils_training.predictions_models_data([model], x_test, batch_size=400)


x_test = data_doodle.df_to_image_array_doodle(test_df.loc[20000:((20000+20000)-1), :], size_imgs, 
                                                    transforms=test_transforms, norm=norm, 
                                                    color=color, trazos=trazos)
test_predictions_second = utils_training.predictions_models_data([model], x_test, batch_size=400)


x_test = data_doodle.df_to_image_array_doodle(test_df.loc[40000:((40000+20000)-1), :], size_imgs, 
                                                    transforms=test_transforms, norm=norm, 
                                                    color=color, trazos=trazos)
test_predictions_third = utils_training.predictions_models_data([model], x_test, batch_size=400)


x_test = data_doodle.df_to_image_array_doodle(test_df.loc[60000:((60000+20000)-1), :], size_imgs, 
                                                    transforms=test_transforms, norm=norm, 
                                                    color=color, trazos=trazos)
test_predictions_fourth = utils_training.predictions_models_data([model], x_test, batch_size=400)



x_test = data_doodle.df_to_image_array_doodle(test_df.loc[80000:((80000+15000)-1), :], size_imgs, 
                                                    transforms=test_transforms, norm=norm, 
                                                    color=color, trazos=trazos)
test_predictions_fith = utils_training.predictions_models_data([model], x_test, batch_size=400)



x_test = data_doodle.df_to_image_array_doodle(test_df.loc[(80000+15000):, :], size_imgs, 
                                                    transforms=test_transforms, norm=norm, 
                                                    color=color, trazos=trazos)
test_predictions_six = utils_training.predictions_models_data([model], x_test, batch_size=400)


test_predictions = np.concatenate((test_predictions_first, test_predictions_second, test_predictions_third, test_predictions_fourth, test_predictions_fith, test_predictions_six))
submission = utils_doodle.preds2submission(test_predictions, test_df)

sample2see = 0
x_test = data_doodle.df_to_image_array_doodle(test_df.loc[0:1, :], size_imgs,transforms=test_transforms, norm=norm, color=color, trazos=[3])
plt.title("Reconocemos: " +  submission.iloc[sample2see]["word"])
plt.imshow(x_test[0,-1,:,:].data.cpu().numpy().reshape(size_imgs,size_imgs), cmap="gray")



submission.to_csv('submissions/CE3_'+model_type+"_"+block_type+str(model_cfg)+'_IMG'+str(size_imgs)+da+'.csv', index=False)

