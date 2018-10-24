import torch
import numpy
from skimage.transform import rotate as imrotate

def hflip(img,aux):
    return torch.index_select(img,2,aux)

def Crop(img,i,j,th,tw):
    return img[:,i:i+th,j:j+tw]

def Zscore(img,mean,std):
        return (img-mean)/std

def rotate(img,angle):#unsupported the rest of torchvision options
    bool_cuda=False
    if img.is_cuda:
        bool_cuda=True
        img=img.cpu()
    img=img.numpy()
    img=numpy.transpose(img,(1,2,0))
    img=imrotate(img,numpy.float32(angle),mode='symmetric').astype(numpy.float32)
    img=numpy.transpose(img,(2,0,1))
    if bool_cuda:
        return torch.from_numpy(img).cuda()
    else:
        return torch.from_numpy(img)

