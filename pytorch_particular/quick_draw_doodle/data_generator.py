import os
import cv2
import pandas as pd
import json

import pytorchlib.pytorch_data.transforms as custom_transforms
import pytorchlib.pytorch_library.utils_particular as utils_particular
import pytorchlib.pytorch_library.utils_training as utils_training

BASE_SIZE = 256

def draw_cv2(raw_strokes, size=256, lw=6, time_color=True):
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

def image_generator_doodle(size, batch_size, ks, transforms=[], norm="", lw=6, time_color=True, num_classes=340):
    while True:
        for k in np.random.permutation(ks):
            filename = os.path.join(DP_DIR, 'train_k{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                df['drawing'] = df['drawing'].apply(json.loads)
                x = np.zeros((len(df), size, size, 1))
                for i, raw_strokes in enumerate(df.drawing.values):
                    x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw,
                                             time_color=time_color)

                #x = preprocess_input(x).astype(np.float32)
                xt = []
                if transforms!=[]:
                    for indx, (sample) in enumerate(x):
                        for transform in transforms:
                            sample = custom_transforms.apply_albumentation(transform, sample)
                        xt.append(sample)
                if xt!=[]: x = np.array(xt)
                x = torch.from_numpy(x)
                x = x.permute(0,3,1,2) # Necesitamos los canales en la segunda posicion

                y = torch.from_numpy(utils_training.to_categorical(df.y, num_classes=num_classes))
                # Normalizamos los datos
                if norm != "":
                    current_features = custom_transforms.single_normalize(current_features, norm)
                yield x, y

def df_to_image_array_doodle(df, size, lw=6, time_color=True, transforms=[], norm=""):
    df['drawing'] = df['drawing'].apply(json.loads)
    x = np.zeros((len(df), size, size, 1))
    for i, raw_strokes in enumerate(df.drawing.values):
        x[i, :, :, 0] = draw_cv2(raw_strokes, size=size, lw=lw, time_color=time_color)
    #x = preprocess_input(x).astype(np.float32)
    xt = []
    if transforms!=[]:
        for indx, (sample) in enumerate(x):
            for transform in transforms:
                sample = custom_transforms.apply_albumentation(transform, sample)
            xt.append(sample)
    if xt!=[]: x = np.array(xt)
    x = torch.from_numpy(x)
    x = x.permute(0,3,1,2) # Necesitamos los canales en la segunda posicion
    # Normalizamos los datos
    if norm != "":
        current_features = custom_transforms.single_normalize(current_features, norm)
    return x