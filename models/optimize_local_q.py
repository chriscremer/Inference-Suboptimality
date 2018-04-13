






import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import lognormal2 as lognormal
from utils import lognormal333

from utils import log_bernoulli

import time

import pickle

quick = 0






def optimize_local_q_dist(logposterior, hyper_config, x, q):

    B = x.size()[0] #batch size
    P = 50

    z_size = hyper_config['z_size']
    x_size = hyper_config['x_size']
    if torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
        
    mean = Variable(torch.zeros(B, z_size).type(dtype), requires_grad=True)
    logvar = Variable(torch.zeros(B, z_size).type(dtype), requires_grad=True)

    params = [mean, logvar]
    for aaa in q.parameters():
        params.append(aaa)


    optimizer = optim.Adam(params, lr=.001)

    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 999999):

        # #Sample
        # eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        # logqz = lognormal(z, mean, logvar) #[P,B]

        # fsadfad
        # z, logqz = q.sample(...)
        z, logqz = q.sample(mean, logvar, P)

        logpx = logposterior(z)

        optimizer.zero_grad()


        loss = -(torch.mean(logpx-logqz))
        loss_np = loss.data.cpu().numpy()
        # print (epoch, loss_np)
        # fasfaf

        loss.backward()
        optimizer.step()

        last_100.append(loss_np)
        if epoch % 100 ==0:

            last_100_avg = np.mean(last_100)
            if last_100_avg< best_last_100_avg or best_last_100_avg == -1:
                consecutive_worse=0
                best_last_100_avg = last_100_avg
            else:
                consecutive_worse +=1 
                # print(consecutive_worse)
                if consecutive_worse> 10:
                    # print ('done')
                    break

            if epoch % 2000 ==0:
                print (epoch, last_100_avg, consecutive_worse)#,mean)
            # print (torch.mean(logpx))

            last_100 = []



    # Compute VAE and IWAE bounds

    # #Sample
    # eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
    # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
    # logqz = lognormal(z, mean, logvar) #[P,B]
    z, logqz = q.sample(mean, logvar, 5000)

    # print (logqz)
    # fad
    logpx = logposterior(z)

    elbo = logpx-logqz #[P,B]
    vae = torch.mean(elbo)

    max_ = torch.max(elbo, 0)[0] #[B]
    elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    iwae = torch.mean(elbo_)

    return vae, iwae












