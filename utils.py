import torch
import numpy as np
import h5py

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsample.datasets import TensorDataset
from torch.nn.init import xavier_uniform

from machinedesign.viz import grid_of_images, vert_merge, horiz_merge

def load_data(data_path, image_size, data_type):
    transform=transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    if data_type == 'npy':
        data = np.load(data_path)
        X = torch.from_numpy(data['X']).float()
        if 'y' in data:
            y  = torch.from_numpy(data['y'])
        else:
            y = torch.zeros(len(X))
        X /= X.max()
        dataset = TensorDataset(
            inputs=X, 
            targets=y,
        )

    elif data_type == 'h5':
        data = h5py.File(data_path, 'r')
        X = data['X']
        if 'y' in data:
            y  = (data['y'])
        else:
            y = np.zeros(len(X))
        dataset = H5Dataset(X, y, transform=lambda u:u.float()/255.)

    elif data_type == 'image_folder':
        dataset = datasets.ImageFolder(
            root=data_path,
            transform=transform
        )
    else:
        raise ValueError(data_type)
    return dataset


def preprocess(x):
    return 2.0 * x - 1.0

def deprocess(x):
    return (x + 1) / 2.0

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        xavier_uniform(m.weight.data)
        m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class H5Dataset:

    def __init__(self, X, y, transform=lambda x:x):
        self.X = X
        self.y = y
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(torch.from_numpy(self.X[index])), self.y[index]

    def __len__(self):
        return len(self.X)
