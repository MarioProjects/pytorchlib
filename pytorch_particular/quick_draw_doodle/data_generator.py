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

def draw_cv2_trazos(raw_strokes, size=256, lw=6, time_color=True, color=False, trazos=0, trazo_actual=0):

    # Calculo el numero de trazos totales del dibujo
    trazos_totales = 0
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            trazos_totales+=1
    # Calculo los puntosque le tocan a esta porcion
    puntos_por_canal = int(trazos_totales/trazos)
    inicio_puntos = trazo_actual*puntos_por_canal
    fin_puntos = inicio_puntos + puntos_por_canal

    if color: img = np.zeros((BASE_SIZE, BASE_SIZE, 3), np.uint8)
    else: img = np.zeros((BASE_SIZE, BASE_SIZE), np.uint8)
    
    trazos_actuales = 0
    for t, stroke in enumerate(raw_strokes):
        for i in range(len(stroke[0]) - 1):
            if(trazos_actuales>=inicio_puntos and trazos_actuales<fin_puntos):
                if color:
                    color = COLORS[min(t, len(COLORS)-1)]
                    _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                                    (stroke[0][i + 1], stroke[1][i + 1]), color, lw, lineType=cv2.LINE_AA)
                else:
                    color = 255 - min(t, 10) * 13 if time_color else 255
                    _ = cv2.line(img, (stroke[0][i], stroke[1][i]),
                                (stroke[0][i + 1], stroke[1][i + 1]), color, lw)
            trazos_actuales+=1
    if size != BASE_SIZE:
        return cv2.resize(img, (size, size))
    else:
        return img

def image_generator_doodle(size, batch_size, ks, data_amount, transforms=[], norm="", lw=6, time_color=True, num_classes=340, color=False, trazos=[0]):
    # data_amount = "1k" "10k" "50k"
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(SHUFFLED_CSVs_DIR + data_amount + "/", 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                df['drawing'] = df['drawing'].apply(json.loads)

                if color and trazos[0]!=0:
                    total_trazos = np.array(trazos).sum()
                    x = np.zeros((len(df), size, size, (total_trazos+1)*3))

                    for i, raw_strokes in enumerate(df.drawing.values):
                        trazos_dibujados = 0
                        for indx, trazos_actuales in enumerate(trazos):
                            for trazo_actual in range(trazos_actuales):
                                x[i, :, :, trazos_dibujados:(trazos_dibujados+3)] = draw_cv2_trazos(raw_strokes, size=size, lw=lw, time_color=time_color, color=color, trazos=trazos_actuales, trazo_actual=trazo_actual)
                                trazos_dibujados+=3
                            if (trazos_dibujados+3)==(total_trazos+1)*3:
                                x[i, :, :, trazos_dibujados:(trazos_dibujados+3)] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)
                                trazos_dibujados+=3

                elif color:
                    x = np.zeros((len(df), size, size, 3))
                    for i, raw_strokes in enumerate(df.drawing.values):
                        x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)
                elif trazos[0]!=0:

                    total_trazos = np.array(trazos).sum()
                    x = np.zeros((len(df), size, size, total_trazos+1))

                    for i, raw_strokes in enumerate(df.drawing.values):
                        trazos_dibujados = 0
                        for indx, trazos_actuales in enumerate(trazos):
                            for trazo_actual in range(trazos_actuales):
                                x[i, :, :, trazos_dibujados] = draw_cv2_trazos(raw_strokes, size=size, lw=lw, time_color=time_color, color=color, trazos=trazos_actuales, trazo_actual=trazo_actual)
                                trazos_dibujados+=1
                            if trazos_dibujados==total_trazos:
                                x[i, :, :, trazos_dibujados] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)
                                trazos_dibujados+=1
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


def df_to_image_array_doodle(df, size, lw=6, time_color=True, transforms=[], norm="", color=False, trazos=[0]):
    df=df.copy(deep=True)
    df['drawing'] = df['drawing'].apply(json.loads)

    if color and trazos[0]!=0:
        total_trazos = np.array(trazos).sum()
        x = np.zeros((len(df), size, size, (total_trazos+1)*3))

        for i, raw_strokes in enumerate(df.drawing.values):
            trazos_dibujados = 0
            for indx, trazos_actuales in enumerate(trazos):
                for trazo_actual in range(trazos_actuales):
                    x[i, :, :, trazos_dibujados:(trazos_dibujados+3)] = draw_cv2_trazos(raw_strokes, size=size, lw=lw, time_color=time_color, color=color, trazos=trazos_actuales, trazo_actual=trazo_actual)
                    trazos_dibujados+=3
                if (trazos_dibujados+3)==(total_trazos+1)*3:
                    x[i, :, :, trazos_dibujados:(trazos_dibujados+3)] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)
                    trazos_dibujados+=3

    elif color:
        x = np.zeros((len(df), size, size, 3))
        for i, raw_strokes in enumerate(df.drawing.values):
            x[i, :, :, :] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)

    elif trazos[0]!=0:

        total_trazos = np.array(trazos).sum()
        x = np.zeros((len(df), size, size, total_trazos+1))

        for i, raw_strokes in enumerate(df.drawing.values):
            trazos_dibujados = 0
            for indx, trazos_actuales in enumerate(trazos):
                for trazo_actual in range(trazos_actuales):
                    x[i, :, :, trazos_dibujados] = draw_cv2_trazos(raw_strokes, size=size, lw=lw, time_color=time_color, color=color, trazos=trazos_actuales, trazo_actual=trazo_actual)
                    trazos_dibujados+=1
                if trazos_dibujados==total_trazos:
                    x[i, :, :, trazos_dibujados] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color, color=color)
                    trazos_dibujados+=1

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

