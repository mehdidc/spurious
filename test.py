import gan
from gan import * # NOQA
from clize import run

def train():
    nb_colors = 1
    image_size = 28
    model = {
        'generator': {
            'name': 'GenMnist',
            'params': {
                #'latent_size': 128,
                #'nb_gen_filters': 128,
                #'nb_colors': nb_colors,
                #'image_size': image_size
            }
        },
        'discriminator': {
            'name': 'DiscrMnist',

            'params': {
                #'nb_discr_filters': 128,
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
        'output_folder': 'out',
    }
    gan.train(params)


def generate():
    params = {
        'folder': 'out',
        'nb_samples': 1000,
        'output_file': 'out/gen.npz'
    }
    gan.generate(params)

def reconstruct():
    params = {
        'folder': 'out',
        'data': {
            'path': 'data/celeba64.npz',
            'type': 'npy',
            'image_size': 64,
            'nb_colors': 3,
        },
        'batch_size': 64,
        'nb_batches': 1,
        'lr': 100.0,
        'nb_iter': 200,
        'output_file': 'out/recons.npz'
    }
    gan.reconstruct(params)


if __name__ == '__main__':
    run([train, generate, reconstruct])
