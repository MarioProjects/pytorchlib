import os
import cv2
import pandas as pd
import json
import numpy as np

import torch

import pytorchlib.pytorch_data.load_data as load_data
import pytorchlib.pytorch_library.utils_training as utils_training

BASE_SIZE = 256
SHUFFLED_CSVs_DIR = "/home/maparla/DeepLearning/KaggleDatasets/quick_draw_doodle/shuffle-csvs"
COLORS = [(255, 0, 0) , (255, 255, 0),  (128, 255, 0),  (0, 255, 0), (0, 255, 128), (0, 255, 255),
          (0, 128, 255), (0, 0, 255), (128, 0, 255), (255, 0, 255)]



def draw_cv2(raw_strokes, size=256, lw=6, time_color=True, color=False):
    if color:
        img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = COLORS[min(t, len(COLORS)-1)]
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                                (stroke[0][i + 1], stroke[1][i + 1]), color, lw, lineType=cv2.LINE_AA)
        if size != BASE_SIZE:
            return cv2.resize(img, (size, size))
        else:
            return img
    else:
        img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
        for t, stroke in enumerate(raw_strokes):
            for i in range(len(stroke[0]) - 1):
                color = 255 - min(t, 10) * 13 if time_color else 255
                _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                            (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
        if size != BASE_SIZE:
            return cv2.resize(img, (size, size))
        else:
            return img

def image_generator_doodle(size, batch_size, ks, data_amount, transforms=[], norm="", lw=6, time_color=True, num_classes=340, color=False):
    # data_amount = "1k" "10k" "50k"
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(SHUFFLED_CSVs_DIR + data_amount + "/", 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                df['drawing'] = df['drawing'].apply(json.loads)
                if color:
                    x = np.zeros((len(df), size, size, 3))
                    for i, raw_strokes in enumerate(df.drawing.values):
                        x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)
                else:
                    x = np.zeros((len(df), size, size, 1))
                    for i, raw_strokes in enumerate(df.drawing.values):
                        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)

                x = x.astype(np.float32)
                xt = []
                if transforms!=[]:
                    for indx, (sample) in enumerate(x):
                        for transform in transforms:
                            sample = load_data.apply_img_albumentation(transform, sample)
                        sample = np.array(sample)
                        if color: xt.append(sample.reshape(sample.shape[0], sample.shape[1], 3))
                        else: xt.append(sample.reshape(sample.shape[0], sample.shape[1], 1))
                if xt!=[]: x = np.array(xt)
                x = torch.from_numpy(x)
                x = x.permute(0,3,1,2) # Necesitamos los canales en la segunda posicion

                #y = torch.from_numpy(utils_training.to_categorical(df.y, num_classes=num_classes))
                y = torch.from_numpy(np.array(df.y)) # Crossentropy de Pytorch no trabaja one hot!
                # Normalizamos los datos
                if norm != "":
                    x = load_data.single_normalize(x, norm)
                yield x, y


def df_to_image_array_doodle(df, size, lw=6, time_color=True, transforms=[], norm="", color=False):
    df['drawing'] = df['drawing'].apply(json.loads)

    if color:
        x = np.zeros((len(df), size, size, 3))
        for i, raw_strokes in enumerate(df.drawing.values):
            x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)
    else:
        x = np.zeros((len(df), size, size, 1))
        for i, raw_strokes in enumerate(df.drawing.values):
            x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)


    x = x.astype(np.float32)
    xt = []
    if transforms!=[]:
        for indx, (sample) in enumerate(x):
            for transform in transforms:
                sample = load_data.apply_img_albumentation(transform, sample)
            #xt.append(sample)
            sample = np.array(sample)
            if color: xt.append(sample.reshape(sample.shape[0], sample.shape[1], 3))
            else: xt.append(sample.reshape(sample.shape[0], sample.shape[1], 1))
    if xt!=[]: x = np.array(xt)
    x = torch.from_numpy(x)
    x = x.permute(0,3,1,2) # Necesitamos los canales en la segunda posicion
    # Normalizamos los datos
    if norm != "":
        x = load_data.single_normalize(x, norm)
    return x