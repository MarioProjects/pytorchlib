import torch
from torch import nn, optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2

class MiMicNetCombined(nn.Module):
    def __init__(self, mimicOriginal, addedOut):
        super(MiMicNetCombined, self).__init__()
        self.net_type = mimicOriginal.net_type

        self.orginal_net = nn.Sequential(mimicOriginal)
        self.out = nn.Sequential(addedOut)


    def forward(self, x):
        out_mimic = self.orginal_net(x)
        # Caso en el que se devuelve reshape y logits
        if type(out_mimic) is list or type(out_mimic) is tuple:
            reshape_mimic = out_mimic[0]
        else: reshape_mimic = out_mimic
        out = self.out(reshape_mimic)
        return reshape_mimic, out


def strokes_to_img(in_strokes):
    # Para transformar vectores con puntos que representan lineas a imagenes
    # util de competicion quick_draw_doodle: https://www.kaggle.com/c/quickdraw-doodle-recognition
    in_strokes = eval(in_strokes)
    # make an agg figure
    fig, ax = plt.subplots()
    for x,y in in_strokes:
        ax.plot(x, y, linewidth=12., color="black") #  marker='.',
    ax.axis('off')
    fig.canvas.draw()

    # grab the pixel buffer and dump it into a numpy array
    X = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)

    return (cv2.resize(X, (256, 256)) / 255.)[::-1]