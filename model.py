import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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

class VAE(nn.Module):

    def __init__(self, nb_colors=1, nb_filters=64, latent_size=256, image_size=64, use_cuda=True):
        super().__init__()
        self.nb_colors = nb_colors
        self.nb_filters = nb_filters
        self.latent_size = latent_size
        self.image_size = image_size
        self.use_cuda = use_cuda

        nc = self.nb_colors
        nf = self.nb_filters
        w = self.image_size

        nb_blocks = int(np.log(w)/np.log(2)) - 3
        layers = [
            nn.Conv2d(nc, nf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        for _ in range(nb_blocks):
            layers.extend([
                nn.Conv2d(nf, nf * 2, 4, 2, 1),
                #nn.BatchNorm2d(nf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            nf = nf * 2
        
        self._encode = nn.Sequential(*layers)

        wl = w // 2**(nb_blocks+1)
        self.pre_latent_size = (nf, wl, wl)
        self.latent = nn.Sequential(
            nn.Linear(nf * wl * wl, latent_size * 2),
        )
        self.post_latent = nn.Sequential(
            nn.Linear(latent_size, nf * wl * wl)
        )
        layers = []
        for _ in range(nb_blocks):
            layers.extend([
                nn.ConvTranspose2d(nf, nf // 2, 4, 2, 1),
                #nn.BatchNorm2d(nf // 2),
                nn.ReLU(True),
            ])
            nf = nf // 2
        layers.append(
            nn.ConvTranspose2d(nf,  nc, 4, 2, 1),
        )
        layers.append(nn.Sigmoid())
        self._decode = nn.Sequential(*layers)

    def forward(self, input):
        h_mu, h_log_var = self.encode(input)
        noise = torch.randn(h_mu.size())
        if self.use_cuda:
            noise = noise.cuda()
        noise = Variable(noise)
        h = h_mu + torch.exp(h_log_var * 0.5) * noise
        return h_mu, h_log_var, self.decode(h)
    
    def encode(self, input):
        x = self._encode(input)
        x = x.view(x.size(0), -1)
        h = self.latent(x)
        h_mu = h[:, 0:self.latent_size]
        h_log_var = h[:, self.latent_size:]
        return h_mu, h_log_var

    def decode(self, h):
        x = self.post_latent(h)
        x = x.view((x.size(0),) + self.pre_latent_size)
        return self._decode(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, nb_colors, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_type = mask_type
        self.nb_colors = nb_colors
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
        for i in range(nb_colors):
            for j in range(i + 1, nb_colors):
                if mask_type == 'A':
                    # all centers of channels are 0
                    # thus, put centers of channels which correspond to j > i to 1
                    # that is, color j can use the values of color i because j > i
                    self.mask[j::nb_colors, i::nb_colors, kH // 2, kW // 2] = 1
                elif mask_type == 'B':
                    # all centers of channels are 1
                    # put the centers of channels which correspond to i < j to 0
                    # that is, color i cannot use the values of color j because i < j
                    self.mask[i::nb_colors, j::nb_colors, kH // 2, kW // 2] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super().forward(x)


class PixelCNN(nn.Module):

    def __init__(self, nb_layers=6, nb_feature_maps=64, filter_size=5, nb_colors=1, image_size=28, dilation=1):
        super().__init__()
        self.nb_layers = nb_layers
        self.nb_feature_maps = nb_feature_maps
        self.filter_size = filter_size
        self.nb_colors = nb_colors
        self.image_size = image_size
        self.dilation = dilation
        
        fm = nb_feature_maps  * nb_colors
        fs = filter_size
        pad = ((fs - 1) // 2) 
        d = (dilation - 1) * ((fs - 1) // 2)
        layers = []
        layers.append(MaskedConv2d('A', nb_colors, nb_colors, fm, fs, 1, pad))
        for i in range(nb_layers - 1):
            layers.append(MaskedConv2d('B', nb_colors, fm, fm, fs, 1, pad + d, dilation=dilation))
            layers.append(nn.ReLU(True))
        layers.append(MaskedConv2d('B', nb_colors, fm, 256 * nb_colors, fs, 1, pad))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



models = {
    'Gen': Gen,
    'GenMnist': GenMnist,
    'Discr': Discr,
    'DiscrMnist': DiscrMnist,
    'VAE': VAE,
    'PixelCNN': PixelCNN,
}
