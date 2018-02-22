import torch
import torch.nn as nn
import numpy as np


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
    def __init__(self):
        super().__init__()
        DIM = 64
        self.latent_size = 128
        preprocess = nn.Sequential(
            nn.Linear(128, 4*4*4*DIM),
            nn.ReLU(True),
        )
        block1 = nn.Sequential(
            nn.ConvTranspose2d(4*DIM, 2*DIM, 5),
            nn.ReLU(True),
        )
        block2 = nn.Sequential(
            nn.ConvTranspose2d(2*DIM, DIM, 5),
            nn.ReLU(True),
        )
        deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)

        self.block1 = block1
        self.block2 = block2
        self.deconv_out = deconv_out
        self.preprocess = preprocess

    def forward(self, input):
        DIM = 64
        input = input.view(input.size(0), -1)
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        #print output.size()
        output = nn.Tanh()(output)
        return output

class DiscrMnist(nn.Module):
    def __init__(self):
        super().__init__()
        DIM = 64
        main = nn.Sequential(
            nn.Conv2d(1, DIM, 5, stride=2, padding=2),
            # nn.Linear(OUTPUT_DIM, 4*4*4*DIM),
            nn.LeakyReLU(True),
            nn.Conv2d(DIM, 2*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.LeakyReLU(True),
            nn.Conv2d(2*DIM, 4*DIM, 5, stride=2, padding=2),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
            # nn.Linear(4*4*4*DIM, 4*4*4*DIM),
            # nn.LeakyReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4*4*4*DIM, 1)

    def forward(self, input):
        DIM = 64
        input = input.view(-1, 1, 28, 28)
        out = self.main(input)
        out = out.view(-1, 4*4*4*DIM)
        out = self.output(out)
        out = out.view(out.size(0), 1)
        return out


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
