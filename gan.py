import os
import time
import random
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable, grad
from torchvision.utils import save_image

from utils import load_data, weights_init, preprocess, deprocess

from model import models

cudnn.benchmark = True

def load(folder):
    gen = torch.load(os.path.join(folder, 'netG.th'))
    discr = torch.load(os.path.join(folder, 'netD.th'))
    return gen, discr


def reconstruct(params):
    folder = params['folder']

    data = params['data']
    data_path = data['path']
    image_size = data['image_size']
    nb_colors = data['nb_colors']
    data_type = data['type']

    batch_size = params['batch_size']
    nb_batches = params['nb_batches']
    lr = params['lr']
    output_file = params['output_file']
    nb_iter = params['nb_iter']

    gen, discr = load(folder)
    latent_size = gen.nz

    dataset = load_data(
        data_path,
        image_size,
        data_type,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=1
    )
    use_cuda = torch.cuda.is_available()
    noise = torch.FloatTensor(batch_size, latent_size, 1, 1).normal_(0, 1)
    if use_cuda:
        noise = noise.cuda()
    input = torch.FloatTensor(batch_size, nb_colors, gen.w, gen.w)
    
    grads = {}
    def save_grads(g):
        grads['h'] = g
    
    z_list = []
    x_list = []
    xrec_list = []
    
    gen.eval()
    for b, data in zip(range(nb_batches), dataloader):

        real_cpu, _ = data
        real_cpu = preprocess(real_cpu)
        if nb_colors == 1:
            real_cpu = real_cpu[:, 0:1] 
        input = Variable(real_cpu)
        noise.normal_(0, 1)
        if use_cuda:
            input = input.cuda()
        
        for it in range(nb_iter):
            noisev = Variable(noise, requires_grad=True)
            noisev.register_hook(save_grads)
            fake = gen(noisev)
            loss = ((fake - input)**2).mean()
            loss.backward()
            print(loss.data[0])
            dnoise = grads['h']
            noise -= lr * dnoise.data

        z_list.append(noise.cpu().numpy())
        x_list.append(real_cpu.cpu().numpy())
        xrec_list.append(fake.data.cpu().numpy())
    z = np.concatenate(z_list, axis=0)
    x = np.concatenate(x_list, axis=0)
    xrec = np.concatenate(xrec_list, axis=0)
    np.savez(output_file, x=x, z=z, xrec=xrec)


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
    data_path = data['path']
    data_type = data['type']

    model = params['model']
    nb_discr_iters = model['nb_discr_iters']
    gradient_penalty_coef = model['gradient_penalty_coef'] 
    gen_name = model['generator']['name']
    gen_params = model['generator']['params']
    discr_name = model['discriminator']['name']
    discr_params = model['discriminator']['params']

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

    gen = models[gen_name](**gen_params)
    discr = models[discr_name](**discr_params)
    gen.apply(weights_init)
    discr.apply(weights_init)

    print(gen)
    print(discr)

    input = torch.FloatTensor(batch_size, nb_colors, image_size, image_size)
    noise = torch.FloatTensor(batch_size, gen.latent_size, 1, 1)
    fixed_noise = torch.FloatTensor(batch_size, gen.latent_size, 1, 1).normal_(0, 1)
    if use_cuda:
        gen = gen.cuda()
        discr = discr.cuda()
        input = input.cuda()
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
            Diters = nb_discr_iters
            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, _ = data
                real_cpu = preprocess(real_cpu)
                if nb_colors == 1:
                    real_cpu = real_cpu[:, 0:1]
                discr.zero_grad()
                batch_size = real_cpu.size(0)

                if use_cuda:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)
                errD_real = discr(inputv).mean()

                # train with fake
                noise.resize_(batch_size, gen.latent_size, 1, 1).normal_(0, 1)
                noisev = Variable(noise, volatile=True) # totally freeze netG
                fake = Variable(gen(noisev).data)

                errD_fake = discr(fake).mean()
                
                gp = calc_gradient_penalty(discr, inputv.data, fake.data)
                ws = (errD_real - errD_fake)
                errD = -ws + gradient_penalty_coef * gp
                errD.backward()
                optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            for p in discr.parameters():
                p.requires_grad = False # to avoid computation
            gen.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(batch_size, gen.latent_size, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = gen(noisev)
            errG = -discr(fake).mean() 
            errG.backward()
            optimizerG.step()
            gen_iterations += 1
            
            duration = time.time() - t0
            if nb_updates % 10 == 0:
                print('[{}/{}][{}/{}][{}] Loss_D: {} Loss_G: {} Loss_D_real: {} Loss_D_fake {} WS {} Duration:{}(s)'.format(epoch, nb_epochs, i, len(dataloader), gen_iterations, errD.data[0], errG.data[0],                     errD_real.data[0], errD_fake.data[0], ws.data[0], duration))
            stats = {
                'epoch': epoch,
                'loss_discr': errD.data[0],
                'loss_gen': errG.data[0],
                'loss_discr_real': errD_real.data[0],
                'loss_discr_fake': errD_fake.data[0],
                'ws_dist': ws.data[0],
                'duration': duration,
                'nb_updates': nb_updates,
            }
            stats_list.append(stats)
            nb_updates += 1

        real_cpu = deprocess(real_cpu)
        save_image(real_cpu, '{0}/real_samples.png'.format(output_folder))
        fake = gen(Variable(fixed_noise, volatile=True))
        fake.data = deprocess(fake.data)
        save_image(fake.data, '{0}/fake_samples_{1:03d}.png'.format(output_folder, epoch))

        # do checkpointing
        torch.save(gen, '{0}/netG_epoch_{1:03d}.th'.format(output_folder, epoch))
        torch.save(gen, '{0}/netG.th'.format(output_folder))

        torch.save(discr, '{0}/netD_epoch_{1:03d}.th'.format(output_folder, epoch))
        torch.save(gen, '{0}/netD.th'.format(output_folder))

        pd.DataFrame(stats_list).set_index('nb_updates').to_csv('{}/stats.csv'.format(output_folder))



def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    gradients = grad(
        outputs=disc_interpolates, 
        inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, 
        retain_graph=True, 
        only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
