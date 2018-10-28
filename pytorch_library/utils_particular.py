import torch
from torch import nn, optim

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