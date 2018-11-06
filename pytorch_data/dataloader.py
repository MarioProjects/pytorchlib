import numpy as np
import torch
import torch.utils.data as data
import pytorchlib.pytorch_data.transforms as custom_transforms
import pytorchlib.pytorch_library.utils_particular as utils_particular
import pytorchlib.pytorch_library.utils_training as utils_training

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


def dataloader_from_numpy(features, targets, batch_size, transforms=[], seed=0, norm="", num_classes=0):
    """
    Generador de loaders generico para numpy arrays
    transforms: lista con transformaciones albumentations (LISTA y no Compose!)
    """

    generator = np.random.RandomState(seed=seed)
    pick_order = n_samples = len(features)
    pick_per_epoch = n_samples // batch_size

    #print("""WARNING: This function (dataloader_from_numpy) receives the features with the form [batch, width, height, channels]
    #        and internally transposes these features to [batch, channels, width, height]""")

    while True:  # Infinity loop
        pick_order = generator.permutation(pick_order)
        for i in range(pick_per_epoch):
            current_picks = pick_order[i*batch_size: (i+1)*batch_size]
            current_features = features[current_picks]
            current_targets = targets[current_picks]

            # Debemos aplicar las transformaciones pertinentes definidas en transforms (albumentations)
            current_features_transformed = []
            if transforms!=[]:
                for indx, (sample) in enumerate(current_features):
                    for transform in transforms:
                        sample = custom_transforms.apply_albumentation(transform, sample)
                    current_features_transformed.append(sample)

            # Para evitar problemas con imagenes en blanco y negro (1 canal)
            if current_features_transformed!=[]: current_features = np.array(current_features_transformed)
            if len(current_features.shape) == 3: current_features = np.expand_dims(current_features, axis=3)

            current_features = torch.from_numpy(current_features)
            current_targets = utils_training.to_categorical(current_targets, num_classes=num_classes)
            current_targets = torch.from_numpy(current_targets)

            current_features = current_features.permute(0,3,1,2)

            # Normalizamos los datos
            if norm != "":
                current_features = custom_transforms.single_normalize(current_features, norm)
            yield current_features, current_targets


def gen_quick_draw_doodle(df, pick_order, pick_per_epoch, NAME_TO_CLASS, batch_size, generator, transforms, norm):
    while True:  # Infinity loop
        pick_order = generator.permutation(pick_order)
        for i in range(pick_per_epoch):
            c_pick = pick_order[i*batch_size: (i+1)*batch_size]
            dfs = df.iloc[c_pick]
            out_imgs = list(map(utils_particular.strokes_to_img, dfs["drawing"]))

            current_features = np.array(out_imgs)[:, :, :, :3].astype(np.float32)
            current_targets = np.array([NAME_TO_CLASS[x] for x in dfs["word"]])

            # Debemos aplicar las transformaciones pertinentes definidas en all_augmentations
            current_features_transformed = []
            if transforms!=[]:
                for indx, (sample) in enumerate(current_features):
                    for transform in transforms:
                        sample = custom_transforms.apply_albumentation(transform, sample)
                    current_features_transformed.append(sample)

            # Para evitar problemas con imagenes en blanco y negro (1 canal)
            if current_features_transformed!=[]: current_features = np.array(current_features_transformed)
            if len(current_features.shape) == 3: current_features = np.expand_dims(current_features, axis=3)

            current_features = torch.from_numpy(current_features)
            current_targets = torch.from_numpy(current_targets)
            current_features = current_features.permute(0,3,1,2)

            # Normalizamos los datos
            if norm != "":
                current_features = custom_transforms.single_normalize(current_features, norm)
            yield current_features, current_targets