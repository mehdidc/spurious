import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Gen(nn.Module):
    def __init__(self, latent_size=100, nb_gen_filters=64, nb_colors=1, image_size=64):
        super().__init__()
        self.latent_size = latent_size
        self.nb_gen_filters = nb_gen_filters
        self.nb_colors = nb_colors
        self.image_size = image_size

        nz = self.latent_size
        ngf = self.nb_gen_filters
        w = self.image_size
        nc = self.nb_colors
        
        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ngf * 2**(nb_blocks + 1)
        layers = [
            nn.ConvTranspose2d(nz, nf, 4, 1, 0, bias=False),
            nn.ReLU(True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1, bias=False),
                nn.ReLU(True),
            ]) 
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf, nc, 4, 2, 1, bias=False)
        )
        layers.append(nn.Tanh())
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        out = self.main(input)
        return out

class Discr(nn.Module):

    def __init__(self, nb_colors=1, nb_discr_filters=64, image_size=64):
        super().__init__()

        self.nb_colors = nb_colors
        self.nb_discr_filters = nb_discr_filters
        self.image_size = image_size
        
        w = self.image_size
        ndf = self.nb_discr_filters
        nc = self.nb_colors

        nb_blocks = int(np.log(w)/np.log(2)) - 3
        nf = ndf 
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2
        layers.append(
            nn.Conv2d(nf, 1, 4, 1, 0, bias=False)
        )
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        out = self.main(input)
        return out.view(out.size(0), 1)


class GenMnist(nn.Module):
    def __init__(self, latent_size=100, nb_gen_filters=128):
        super().__init__()
        
        self.nb_gen_filters = nb_gen_filters
        self.latent_size = latent_size

        d = nb_gen_filters
        self.deconv1 = nn.ConvTranspose2d(latent_size, d*8, 7, 1, 0)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv3 = nn.ConvTranspose2d(d*4, 1, 4, 2, 1)

    def forward(self, input):
        x = F.relu((self.deconv1(input)))
        x = F.relu((self.deconv2(x)))
        x = F.tanh(self.deconv3(x))
        return x


class DiscrMnist(nn.Module):

    def __init__(self, nb_discr_filters=128):
        super().__init__()
        d = nb_discr_filters
        self.conv1 = nn.Conv2d(1, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv3 = nn.Conv2d(d*2, 1, 7, 1, 0)

    def forward(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu((self.conv2(x)), 0.2)
        x = self.conv3(x)
        return x


class LayerNorm(nn.Module):

    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)

        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y

models = {
    'Gen': Gen,
    'GenMnist': GenMnist,
    'Discr': Discr,
    'DiscrMnist': DiscrMnist,
}
