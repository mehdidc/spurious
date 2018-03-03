import os
from clize import run
import gan
import vae
import pixelcnn
from model import *
from utils import load_data

def mnist_gan():
    nb_colors = 1
    image_size = 28
    model = {
        'generator': {
            'name': 'GenMnist',
            'params': {
                'latent_size': 100,
                'nb_gen_filters': 128,
            }
        },
        'discriminator': {
            'name': 'DiscrMnist',

            'params': {
                'nb_discr_filters': 128,
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
            'train':{
                'path': 'data/digits.npz',
                'type': 'npy',
                'image_size': image_size,
                'nb_colors': nb_colors,
            },
            'test':{
                'path': 'data/digits_test.npz',
                'type': 'npy',
                'image_size': image_size,
                'nb_colors': nb_colors,
            }

        },
        'seed': 42,
        'output_folder': 'mnist',
        'family': 'gan',
    }
    return params


def mnist_pixelcnn():
    params = mnist_gan()
    params['model'] = {
        'name': 'PixelCNN',
        'params':{
            'nb_layers': 6,
            'nb_feature_maps': 64,
            'filter_size': 5,
            'dilation': 2,
        }
    }
    params['output_folder'] = 'results/pixelcnn/mnist'
    params['family'] = 'pixelcnn'
    params['optim']['algo']['params']['lr'] = 1e-3
    return params


def celeba_gan():
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
            'train':{
                'path': 'data/celeba64_align.h5',
                'type': 'h5',
                'image_size': image_size,
                'nb_colors': nb_colors,
            },
            'test':{
                'path': 'data/lfw.npz',
                'type': 'npy',
                'image_size': image_size,
                'nb_colors': nb_colors,
            }
 
        },
        'seed': 42,
        'output_folder': 'celeba',
        'family': 'gan',
    }
    return params


def celeba_vae():
    params = celeba_gan()
    params['model'] = {
        'name': 'VAE',
        'params': {
            'nb_colors': 3,
            'nb_filters': 64,
            'latent_size': 32,
            'image_size': 64,
        }
    }
    params['family'] = 'vae'
    params['output_folder'] = 'results/vae/celeba'
    return params



def cifar_gan():
    image_size = 32
    nb_colors = 3
    params = celeba_gan()
    params['model'] = {
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

    params['data'] = {
        'train':{
            'path': 'data/cifar10.npz',
            'type': 'npy',
            'image_size': image_size,
            'nb_colors': nb_colors,
        },
        'test':{
            'path': 'data/cifar10_test.npz',
            'type': 'npy',
            'image_size': image_size,
            'nb_colors': nb_colors,
        },
    }
    params['output_folder'] = 'cifar'
    return params


def cifar_vae():
    params = cifar_gan()
    params['model'] = {
        'name': 'VAE',
        'params': {
            'nb_colors': 3,
            'nb_filters': 64,
            'latent_size': 512,
            'image_size': 32,
        }
    }
    params['family'] = 'vae'
    params['output_folder'] = 'results/vae/cifar'
    return params


def cifar_pixelcnn():
    params = cifar_gan()
    params['model'] = {
        'nb_layers': 6,
        'nb_feature_maps': 64,
        'filter_size': 5,
    }
    params['output_folder'] = 'results/pixelcnn/cifar'
    params['family'] = 'pixelcnn'
    params['optim']['algo']['params']['lr'] = 1e-3
    return params


    params['family'] = 'vae'
    params['output_folder'] = 'results/vae/cifar'
    return params



def train(model):
    params = globals()[model]()
    family = globals()[params['family']]
    family.train(params)


def generate(folder, *, family='gan'):
    params = {
        'folder': folder,
        'nb_samples': 1000,
        'output_file': '{}/gen.npz'.format(folder),
        'family': family
    }
    family = globals()[params['family']]
    family.generate(params)


def reconstruct(folder, *, nb_examples=None, batch_size=64):
    data = globals()[folder]()['data']['test']
    dataset = load_data(data['path'], data['image_size'], data['type'])
    if nb_examples is None:
        nb_examples = len(dataset)
    nb_examples = int(nb_examples)
    params = {
        'folder': folder,
        'data': data,
        'batch_size': batch_size,
        'nb_batches': nb_examples//batch_size,
        'lr': 100.0,
        'nb_iter': 500,
        'output_file': os.path.join(folder, 'recons.npz')
    }
    family = globals()[params['family']]
    family.reconstruct(params)


if __name__ == '__main__':
    run([train, generate, reconstruct])
