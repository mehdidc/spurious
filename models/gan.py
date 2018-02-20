import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision.utils import save_image


cudnn.benchmark = True


def load(folder):
    gen = torch.load(os.path.join(folder, 'netG.th'))
    discr = torch.load(os.path.join(folder, 'netD.th'))
    return gen, discr

def generate(params):
    folder = params['folder']
    output_file = params['output_file']
    nb_samples = params['nb_samples']
    gen, discr = load(folder)

    latent_size = gen.nz

    use_cuda = torch.cuda.is_available()
    fixed_noise = torch.FloatTensor(nb_samples, latent_size, 1, 1).normal_(0, 1)
    if use_cuda:
        fixed_noise = fixed_noise.cuda()
    fixed_noise = Variable(fixed_noise)
    fake = gen(fixed_noise)
    fake = fake.data.cpu().numpy()
    np.savez(output_file, X=fake)



def train(params):
    seed = params['seed']
    output_folder = params['output_folder']

    data = params['data']
    image_size = data['image_size']
    nb_colors = data['nb_colors']
    data_folder = data['folder']

    model = params['model']
    latent_size = model['latent_size']
    nb_discr_filters = model['nb_discr_filters']
    nb_gen_filters = model['nb_gen_filters']
    nb_extra_layers = model['nb_extra_layers']
    clamp_value = model['clamp_value']

    optim = params['optim']
    algo = getattr(torch.optim, optim['algo']['name'])
    algo_params = optim['algo']['params']
    batch_size = optim['batch_size']
    num_workers = optim['num_workers']
    nb_epochs = optim['nb_epochs']

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('Using cuda...')
    
    normalize_mu = (0.5,) * nb_colors
    normalize_std = (0.5,) * nb_colors
    dataset = datasets.ImageFolder(
        root=data_folder,
        transform=transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mu, normalize_std),
    ])) 
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=num_workers
    )

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    gen = DCGAN_G(image_size, latent_size, nb_colors, nb_gen_filters, n_extra_layers=nb_extra_layers)
    gen.apply(_weights_init)
    discr = DCGAN_D(image_size, latent_size, nb_colors, nb_discr_filters, n_extra_layers=nb_extra_layers)
    discr.apply(_weights_init)

    print(gen)
    print(discr)


    input = torch.FloatTensor(batch_size, nb_colors, image_size, image_size)
    noise = torch.FloatTensor(batch_size, latent_size, 1, 1)
    fixed_noise = torch.FloatTensor(batch_size, latent_size, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        gen = gen.cuda()
        discr = discr.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    
    optimizerD = algo(discr.parameters(), **algo_params)
    optimizerG = algo(gen.parameters(), **algo_params)

    gen_iterations = 0
    nb_updates = 0
    stats_list = []
    for epoch in range(nb_epochs):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            t0 = time.time()
            ############################
            # (1) Update D network
            ###########################
            for p in discr.parameters(): # reset requires_grad
                p.requires_grad = True # they are set to False below in netG update

            # train the discriminator Diters times
            #if gen_iterations < 25 or gen_iterations % 500 == 0:
            #    Diters = 100
            #else:
            #    Diters = nb_discr_filters
            Diters = nb_discr_filters

            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # clamp parameters to a cube
                for p in discr.parameters():
                    p.data.clamp_(-clamp_value, clamp_value)

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, _ = data
                if nb_colors == 1:
                    real_cpu = real_cpu[:, 0:1]

                discr.zero_grad()
                batch_size = real_cpu.size(0)

                if use_cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                errD_real = discr(inputv)
                errD_real.backward(one)

                # train with fake
                noise.resize_(batch_size, latent_size, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile=True) # totally freeze netG
                fake = Variable(gen(noisev).data)
                inputv = fake
                errD_fake = discr(inputv)
                errD_fake.backward(mone)
                errD = errD_real - errD_fake
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in discr.parameters():
                p.requires_grad = False # to avoid computation
            gen.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batch_size, latent_size, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = gen(noisev)
            errG = discr(fake)
            errG.backward(one)
            optimizerG.step()
            gen_iterations += 1
            
            duration = time.time() - t0
            print('[{}/{}][{}/{}][{}] Loss_D: {} Loss_G: {} Loss_D_real: {} Loss_D_fake {} Duration:{}(s)'.format(epoch, nb_epochs, i, len(dataloader), gen_iterations, errD.data[0], errG.data[0],                     errD_real.data[0], errD_fake.data[0], duration))
            stats = {
                'epoch': epoch,
                'loss_discr': errD.data[0],
                'loss_gen': errG.data[0],
                'loss_discr_real': errD_real.data[0],
                'loss_discr_fake': errD_fake.data[0],
                'duration': duration,
                'nb_updates': nb_updates,
            }
            stats_list.append(stats)
            nb_updates += 1

        real_cpu = real_cpu.mul(0.5).add(0.5)
        save_image(real_cpu, '{0}/real_samples.png'.format(output_folder))
        fake = gen(Variable(fixed_noise, volatile=True))
        fake.data = fake.data.mul(0.5).add(0.5)
        save_image(fake.data, '{0}/fake_samples_{1:03d}.png'.format(output_folder, epoch))

        # do checkpointing
        torch.save(gen, '{0}/netG_epoch_{1:03d}.th'.format(output_folder, epoch))
        torch.save(gen, '{0}/netG.th'.format(output_folder))

        torch.save(discr, '{0}/netD_epoch_{1:03d}.th'.format(output_folder, epoch))
        torch.save(gen, '{0}/netD.th'.format(output_folder))

        pd.DataFrame(stats_list).to_csv('{}/stats.csv'.format(output_folder))


def _weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DCGAN_D(nn.Module):
    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0):
        super().__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.isize = isize
        self.nz = nz
        self.nc = nc
        self.ndf = ndf
        self.n_extra_layers = n_extra_layers

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        output = self.main(input)
        output = output.mean(0)
        return output.view(1)

class DCGAN_G(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        self.isize = isize
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.n_extra_layers = n_extra_layers

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output 

class DCGAN_D_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ndf,  n_extra_layers=0):
        super(DCGAN_D_nobn, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        # input is nc x isize x isize
        main.add_module('initial.conv.{0}-{1}'.format(nc, ndf),
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module('initial.relu.{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid.{0}-{1}.conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final.{0}-{1}.conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main


    def forward(self, input):
        output = self.main(input)
        output = output.mean(0)
        return output.view(1)

class DCGAN_G_nobn(nn.Module):
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DCGAN_G_nobn, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial.{0}-{1}.convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial.{0}.relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < isize//2:
            main.add_module('pyramid.{0}-{1}.convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid.{0}.relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}.{1}.conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}.{1}.relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final.{0}-{1}.convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final.{0}.tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output 

if __name__ == '__main__':
    params = {
        'model': {
            'latent_size': 100,
            'nb_discr_filters': 128,
            'nb_gen_filters': 128,
            'nb_extra_layers': 0,
            'nb_discr_iters': 5,
            'clamp_value': 0.01,
        },
        'optim':{
            'algo':{
                'name': 'RMSprop',
                'params':{
                    'lr': 0.00005,
                    #'betas': (0.5, 0.999),
                },
            },
            'batch_size': 64,
            'num_workers': 1,
            'nb_epochs': 10000,
        },
        'data': {
            'folder': '/home/mcherti/work/data/mnist/img_classes',
            'image_size': 32,
            'nb_colors': 1,
        },
        'seed': 42,
        'output_folder': 'out',
    }
    train(params)
