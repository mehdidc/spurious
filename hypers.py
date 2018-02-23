import os
from clize import run
import gan
from gan import * # NOQA


from utils import load_data

def mnist():
    nb_colors = 1
    image_size = 28
    model = {
        'generator': {
            'name': 'GenMnist',
            'params': {
                'latent_size': 100,
                'nb_gen_filters': 128,
                #'nb_colors': nb_colors,
                #'image_size': image_size
            }
        },
        'discriminator': {
            'name': 'DiscrMnist',

            'params': {
                'nb_discr_filters': 128,
                #'nb_colors': nb_colors,
                #'image_size': image_size
            }
        },
        'gradient_penalty_coef': 10.0,
        'nb_discr_iters': 5,
    }
    params = {
        'model': model,
        'optim':{
            'algo':{
                'name': 'Adam',
                'params':{
                    'lr': 0.0001,
                    'betas': (0.5, 0.9),
                },
            },
            'batch_size': 64,
            'num_workers': 1,
            'nb_epochs': 10000,
        },
        'data': {
            'path': 'data/digits.npz',
            'type': 'npy',
            'image_size': image_size,
            'nb_colors': nb_colors,
        },
        'seed': 42,
        'output_folder': 'mnist',
    }
    return params

def celeba():
    nb_colors = 3
    image_size = 64
    model = {
        'generator': {
            'name': 'Gen',
            'params': {
                'latent_size': 100,
                'nb_gen_filters': 128,
                'nb_colors': nb_colors,
                'image_size': image_size
            }
        },
        'discriminator': {
            'name': 'Discr',
            'params': {
                'nb_discr_filters': 128,
                'nb_colors': nb_colors,
                'image_size': image_size
            }
        },
        'gradient_penalty_coef': 10.0,
        'nb_discr_iters': 5,
    }
    params = {
        'model': model,
        'optim':{
            'algo':{
                'name': 'Adam',
                'params':{
                    'lr': 0.0001,
                    'betas': (0.5, 0.9),
                },
            },
            'batch_size': 64,
            'num_workers': 1,
            'nb_epochs': 10000,
        },
        'data': {
            'path': 'data/celeba64_align.h5',
            'type': 'h5',
            'image_size': image_size,
            'nb_colors': nb_colors,
        },
        'seed': 42,
        'output_folder': 'celeba',
    }
    return params


def train(model):
    params = globals()[model]()
    gan.train(params)

def generate(folder):
    params = {
        'folder': folder,
        'nb_samples': 1000,
        'output_file': '{}/gen.npz'.format(folder)
    }
    gan.generate(params)

def reconstruct(folder):
    data = globals()[folder]()['data']
    dataset = load_data(data['data_path'], data['image_size'], data['data_type'])
    batch_size = 64
    params = {
        'folder': folder,
        'data': data,
        'batch_size': batch_size,
        'nb_batches': len(dataset)//batch_size,
        'lr': 100.0,
        'nb_iter': 300,
        'output_file': os.path.join(folder, 'recons.npz')
    }
    gan.reconstruct(params)

if __name__ == '__main__':
    run([train, generate, reconstruct])
