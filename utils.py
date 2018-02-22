import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchsample.datasets import TensorDataset
from torch.nn.init import xavier_uniform

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


