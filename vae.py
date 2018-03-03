import time
import random
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from utils import load_data, weights_init, grid_of_images, vert_merge
from skimage.io import imsave

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

    ae = models[model_name](use_cuda=use_cuda, **model_params)
    ae.apply(weights_init)

    input = torch.FloatTensor(batch_size, nb_colors, image_size, image_size)
    fixed_noise = torch.FloatTensor(batch_size, ae.latent_size).normal_(0, 1)
    if use_cuda:
        ae = ae.cuda()
        input = input.cuda()
        fixed_noise = fixed_noise.cuda()

    optimizer = algo(ae.parameters(), **algo_params)
 
    input = Variable(input)
    stats_list = []
    nb_updates = 0
    for epoch in range(nb_epochs):
        for i, (x, _) in enumerate(dataloader):
            t0 = time.time()
            if use_cuda:
                x = x.cuda()
            x = Variable(x)
            h_mu, h_log_var, xrec = ae(x)
            recons_error, kl = ELBO(x, xrec, h_mu, h_log_var)
            loss = (recons_error + kl)
            ae.zero_grad()
            loss.backward()
            optimizer.step()
            duration = time.time() - t0
            if nb_updates % 10 == 0:
                print('[{}/{}][{}/{}][{}] Loss: {:.6f} Rec: {:.6f} KL: {:.6f} Duration:{:.6f}(s)'.format(epoch, nb_epochs, i, len(dataloader), nb_updates, loss.data[0], recons_error.data[0], kl.data[0], duration))
            stats = {
                'epoch': epoch,
                'loss': loss.data[0],
                'recons_error': recons_error.data[0],
                'kl': kl.data[0],
                'duration': duration,
                'nb_updates': nb_updates,
            }
            stats_list.append(stats)
            nb_updates += 1
        
        # recons
        im1 = grid_of_images(x.data.cpu().numpy(), shape=(len(x), 1), normalize=True)
        im2 = grid_of_images(xrec.data.cpu().numpy(), shape=(len(xrec), 1), normalize=True)
        im = vert_merge(im1, im2)
        imsave('{0}/recons/{1:03d}.png'.format(output_folder, epoch), im)

        # generate
        fixed_noise_v = Variable(fixed_noise)
        xrec = ae.decode(fixed_noise_v)
        im = grid_of_images(xrec.data.cpu().numpy(), normalize=True)
        imsave('{0}/gen/{1:03d}.png'.format(output_folder, epoch), im)

        # do checkpointing
        torch.save(ae, '{0}/ae.th'.format(output_folder))
        pd.DataFrame(stats_list).set_index('nb_updates').to_csv('{}/stats.csv'.format(output_folder))


# Reconstruction + KL divergence losses summed over all elements and batch
def ELBO(x, xr, mu, log_var):
    x = x.view(x.size(0), -1)
    xr = xr.view(xr.size(0), -1)
    recons_error = (((x - xr) ** 2).sum(1)).mean()
    kl = (-0.5 * torch.sum(1 + log_var - mu ** 2 - torch.exp(log_var), 1)).mean()
    return recons_error, kl
