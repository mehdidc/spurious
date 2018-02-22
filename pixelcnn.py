import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchsample.datasets import TensorDataset
import torch.nn.functional as F

from utils import weights_init, load_data, preprocess, deprocess

cudnn.benchmark = True

class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

class Net(nn.Module):

    def __init__(self, nb_layers, nb_feature_maps, filter_size, nb_colors):
        super().__init__()
        self.nb_layers = nb_layers
        self.nb_feature_maps = nb_feature_maps
        self.filter_size = filter_size
        self.nb_colors = nb_colors
        
        fm = nb_feature_maps 
        fs = filter_size
        pad = (fs - 1) // 2
        layers = []
        
        for i in range(nb_layers):
            layers.append(MaskedConv2d('A' if i == 0 else 'B', nb_colors if i == 0 else fm, fm, fs, 1, pad, bias=False))
            layers.append(nn.BatchNorm2d(fm))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(fm, 256 * nb_colors, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train(params):
    seed = params['seed']
    output_folder = params['output_folder']

    data = params['data']
    image_size = data['image_size']
    nb_colors = data['nb_colors']
    data_path = data['path']
    data_type = data['type']

    model = params['model']
    nb_layers = model['nb_layers']
    nb_feature_maps = model['nb_feature_maps']
    filter_size = model['filter_size']

    optim = params['optim']
    algo = getattr(torch.optim, optim['algo']['name'])
    algo_params = optim['algo']['params']
    batch_size = optim['batch_size']
    num_workers = optim['num_workers']
    nb_epochs = optim['nb_epochs']


    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using cuda...')
  
    dataset = load_data(
        data_path,
        image_size,
        data_type,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = Net(nb_layers, nb_feature_maps, filter_size, nb_colors)
    gen.apply(weights_init)

    input = torch.FloatTensor(batch_size, nb_colors, image_size, image_size)
    if use_cuda:
        gen = gen.cuda()
        input = input.cuda()

    optimizer = algo(gen.parameters(), **algo_params)
    nb_updates = 0
    stats_list = []
    for epoch in range(nb_epochs):
        for X, _ in dataloader:
            X = preprocess(X)
            if nb_colors == 1:
                X = X[:, 0:1]
            
            Xtarget = Variable((X * 255).long())
            Xtarget = Xtarget.permute(0, 2, 3, 1)
            Xtarget = Xtarget.contiguous()
            Xtarget = Xtarget.view(-1)

            Xinput = Variable(X)
            if use_cuda:
                Xtarget = Xtarget.cuda()
                Xinput = Xinput.cuda()

            gen.zero_grad()
            Xr = gen(Xinput)
            Xr = Xr.permute(0, 2, 3, 1)
            Xr = Xr.contiguous()
            Xr = Xr.view(Xr.size(0) * Xr.size(1) * Xr.size(2), 256)
            print(Xr.size(), Xtarget.size())
            loss = F.cross_entropy(Xr, Xtarget)
            loss.backward()
            optimizer.step()
            nb_updates += 1
