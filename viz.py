import os
import numpy as np
from skimage.io import imsave
from clize import run
from utils import grid_of_images, vert_merge

def reconstruct(folder, *, out='out.png', nb=None):
    filename = os.path.join(folder, 'recons.npz')
    data = np.load(filename)
    x = data['x']
    xr = data['xrec']
    if nb is not None:
        nb = int(nb)
        x = x[0:nb]
        xr = xr[0:nb]
    im1 = grid_of_images(x, shape=(len(x), 1), normalize=True)
    im2 = grid_of_images(xr, shape=(len(xr), 1), normalize=True)
    im = vert_merge(im1, im2)
    imsave(out, im)

if __name__ == '__main__':
    run([reconstruct])
