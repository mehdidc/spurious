import os
import time
import random
import numpy as np
import pandas as pd
from skimage.io import imsave

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import save_image
from torchsample.datasets import TensorDataset
import torch.nn.functional as F

from utils import weights_init, load_data, preprocess, deprocess, grid_of_images, vert_merge, grid_of_images_with_border
from model import models

cudnn.benchmark = True

def train(params):
    seed = params['seed']
    output_folder = params['output_folder']

    data = params['data']['train']
    image_size = data['image_size']
    nb_colors = data['nb_colors']
    data_path = data['path']
    data_type = data['type']

    model = params['model']
    model_name = model['name']
    model_params = model['params']

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

    gen = models[model_name](**model_params)
    gen.apply(weights_init)

    input = torch.FloatTensor(batch_size, nb_colors, image_size, image_size)
    if use_cuda:
        gen = gen.cuda()
        input = input.cuda()

    optimizer = algo(gen.parameters(), **algo_params)
    nb_updates = 0
    stats_list = []
    for epoch in range(nb_epochs):
        for i, (X, _) in enumerate(dataloader):
            t0 = time.time()
            if nb_colors == 1:
                X = X[:, 0:1]
            Xtarget = Variable((X * 255).long())
            Xtarget = Xtarget.permute(0, 2, 3, 1)
            # nb_examples, h, w, nb_colors
            Xtarget = Xtarget.contiguous()
            Xtarget = Xtarget.view(-1)
            # nb_examples * h * w * nb_colors

            Xinput = Variable(X)
            if use_cuda:
                Xtarget = Xtarget.cuda()
                Xinput = Xinput.cuda()

            gen.zero_grad()
            Xr = gen(Xinput)
            # nb_examples, 256 * nb_colors, h, w
            Xr_orig = Xr
            Xr = Xr.permute(0, 2, 3, 1)
            # nb_examples, h, w, 256 * nb_colors
            Xr = Xr.contiguous()
            Xr = Xr.view(Xr.size(0) * Xr.size(1) * Xr.size(2), Xr.size(3) // nb_colors, nb_colors)
            # nb_examples * h * w, 256, nb_colors
            Xr = Xr.permute(0, 2, 1)
            # nb_examples * h * w, nb_colors, 256
            Xr = Xr.contiguous()
            # nb_examples * h * w, nb_colors, 256
            Xr = Xr.view(Xr.size(0) * Xr.size(1), -1)
            # nb_examples * h * w * nb_colors, 256
            loss = F.cross_entropy(Xr, Xtarget)
            loss.backward()
            optimizer.step()
            duration = time.time() - t0

            if nb_updates % 10 == 0:
                print('[{}/{}][{}/{}][{}] Loss: {:.6f} Duration:{:.6f}(s)'.format(epoch, nb_epochs, i, len(dataloader), nb_updates, loss.data[0],  duration))
            stats = {
                'epoch': epoch,
                'loss': loss.data[0],
                'duration': duration,
                'nb_updates': nb_updates,
            }
            stats_list.append(stats)
            nb_updates += 1

        # recons
        x = X.cpu().numpy()
        xr = Xr_orig.data.cpu()
        xr = discretize(xr, nb_colors=nb_colors)
        xr = xr.numpy().astype('float32')
        im1 = grid_of_images(x, shape=(len(x), 1), normalize=True)
        im2 = grid_of_images(xr, shape=(len(xr), 1), normalize=True)
        im = vert_merge(im1, im2)
        imsave('{0}/recons/{1:03d}.png'.format(output_folder, epoch), im)

        # generate
        x = torch.rand(X.size())
        if use_cuda:
            x = x.cuda()
        g = _generate(gen, x)
        im = grid_of_images_with_border(g.cpu().numpy(), normalize=True)
        imsave('{0}/gen/{1:03d}.png'.format(output_folder, epoch), im)

        # do checkpointing
        torch.save(gen, '{0}/gen.th'.format(output_folder))
        df = pd.DataFrame(stats_list)
        df = df.set_index('nb_updates')
        df.to_csv('{}/stats.csv'.format(output_folder))


def discretize(x, nb_colors=1):
    # assume x is of shape (nb_examples, nb_colors * 256, h, w)
    x = x.view(x.size(0), 256, nb_colors, x.size(2), x.size(3))
    _, x = x.max(1)
    return x


def _generate(model, X):
    nb_colors = X.size(1)
    for h in range(X.size(2)):
        for w in range(X.size(3)):
            for c in range(nb_colors):
                Xv = Variable(X)
                Xr = model(Xv)
                Xr = Xr.view(Xr.size(0), Xr.size(1) // nb_colors, nb_colors, Xr.size(2), Xr.size(3))
                # (nb_examples, 256, 3, h, w)
                p = Xr
                # (nb_examples, 256, 3, h, w)  
                p = p[:, :, c, h, w]
                # (nb_examples, 256)
                p = nn.Softmax(dim=1)(p)
                # (nb_examples, 256)
                p = torch.multinomial(p)
                # (nb_examples,)
                X[:, c, h, w] = p.data.float() / 255.0
    return X


def load(folder):
    return torch.load(os.path.join(folder, 'gen.th'))


def generate(params):
    folder = params['folder']
    output_file = params['output_file']
    nb_samples = params['nb_samples']
    gen = load(folder)

    use_cuda = torch.cuda.is_available()
    x = torch.rand((nb_samples, gen.nb_colors, gen.image_size, gen.image_size))
    if use_cuda:
        gen = gen.cuda()
        x = x.cuda()
    g = _generate(gen, x)
    g = g.cpu().numpy()
    np.savez(output_file, X=g)
