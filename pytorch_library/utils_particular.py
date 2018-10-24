import torch
from torch import nn, optim

class MiMicNetCombined(nn.Module):
    def __init__(self, mimicOriginal, addedOut):
        super(MiMicNetCombined, self).__init__()
        self.net_type = mimicOriginal.net_type

        self.orginal_net = nn.Sequential(mimicOriginal)
        self.out = nn.Sequential(addedOut)


    def forward(self, x):
        reshape_mimic, out_mimic = self.orginal_net(x)
        out = self.out(reshape_mimic)
        return reshape_mimic, out