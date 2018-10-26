import torch.utils.data as data

class ConvolutionalDataset(data.Dataset):

    def __init__(self,feature,label,transforms=None,tag_transforms=None):
        '''Arguments'''
        '''
        *tensors: list with images and labels
        transforms: torchvision.transforms.compose
        '''

        self.images=feature
        self.labels=label

        assert len(self.images.shape)==4, "convolutional datashape must have shape 4"
        assert len(self.labels.shape)==2 , "convolutional targetsshape must have shape 2 aka vector with matrix specifications. Due to sampling in pytorch"
        assert self.labels.shape[1]==1

        self.transforms=transforms
        self.tag_transforms=tag_transforms

    def __getitem__(self,index):

        img, target=self.images[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        if self.tag_transforms is not None:
            target = self.tag_transforms(target)

        return img, target

    def __len__(self):
        return len(self.labels)


