import numpy as np
import torch
import torch.utils.data as data
import pytorchlib.pytorch_data.transforms as custom_transforms
import pytorchlib.pytorch_library.utils_particular as utils_particular
import constants

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


def gen_quick_draw_doodle(df, pick_order, pick_per_epoch, batch_size, generator, transforms, norm):
    while True:  # Infinity loop
        pick_order = generator.permutation(pick_order)
        for i in range(pick_per_epoch):
            c_pick = pick_order[i*batch_size: (i+1)*batch_size]
            dfs = df.iloc[c_pick]
            out_imgs = list(map(utils_particular.strokes_to_img, dfs["drawing"]))

            inputs = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
            labels = np.array([constants.NAME_TO_CLASS[x] for x in dfs["word"]])

            # Debemos aplicar las transformaciones pertinentes definidas en all_augmentations
            for indx, (sample) in enumerate(inputs):
                inputs[indx] = custom_transforms.apply_albumentation(transforms, sample)

            inputs = torch.from_numpy(inputs)
            labels = torch.from_numpy(labels)
            inputs = inputs.permute(0,3,1,2)

            # Normalizamos los datos
            if norm != "":
                inputs = custom_transforms.single_normalize(inputs, norm)
            yield inputs, labels