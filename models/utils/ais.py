




import math
import torch
from torch.autograd import Variable
import numpy as np

from utils import lognormal2 as lognormal
from utils import log_bernoulli

import time


def test_ais(model, data_x, batch_size, display, k, n_intermediate_dists):


    def intermediate_dist(t, z, mean, logvar, zeros, batch):
        logp1 = lognormal(z, mean, logvar)  #[P,B]
        log_prior = lognormal(z, zeros, zeros)  #[P,B]
        log_likelihood = log_bernoulli(model.decode(z), batch)
        logpT = log_prior + log_likelihood
        log_intermediate_2 = (1-float(t))*logp1 + float(t)*logpT
        return log_intermediate_2


    def hmc(z, intermediate_dist_func):

        if torch.cuda.is_available():
            v = Variable(torch.FloatTensor(z.size()).normal_(), volatile=volatile_, requires_grad=requires_grad).cuda()
        else:
            v = Variable(torch.FloatTensor(z.size()).normal_()) 

        v0 = v
        z0 = z

        gradients = torch.autograd.grad(outputs=intermediate_dist_func(z), inputs=z,
                          grad_outputs=grad_outputs,
                          create_graph=True, retain_graph=retain_graph, only_inputs=True)[0]

        gradients = gradients.detach()

        v = v + .5 *step_size*gradients
        z = z + step_size*v

        for LF_step in range(n_HMC_steps):

            # log_intermediate_2 = intermediate_dist(t1, z, mean, logvar, zeros, batch)
            gradients = torch.autograd.grad(outputs=intermediate_dist_func(z), inputs=z,
                              grad_outputs=grad_outputs,
                              create_graph=True, retain_graph=retain_graph, only_inputs=True)[0]
            gradients = gradients.detach()
            v = v + step_size*gradients
            z = z + step_size*v

        # log_intermediate_2 = intermediate_dist(t1, z, mean, logvar, zeros, batch)
        gradients = torch.autograd.grad(outputs=intermediate_dist_func(z), inputs=z,
                          grad_outputs=grad_outputs,
                          create_graph=True, retain_graph=retain_graph, only_inputs=True)[0]
        gradients = gradients.detach()
        v = v + .5 *step_size*gradients

        return z0, v0, z, v


    def mh_step(z0, v0, z, v, step_size, intermediate_dist_func):

        logpv0 = lognormal(v0, zeros, zeros) #[P,B]
        hamil_0 =  intermediate_dist_func(z0) + logpv0
        
        logpvT = lognormal(v, zeros, zeros) #[P,B]
        hamil_T = intermediate_dist_func(z) + logpvT

        accept_prob = torch.exp(hamil_T - hamil_0)

        if torch.cuda.is_available():
            rand_uni = Variable(torch.FloatTensor(accept_prob.size()).uniform_(), volatile=volatile_, requires_grad=requires_grad).cuda()
        else:
            rand_uni = Variable(torch.FloatTensor(accept_prob.size()).uniform_())

        accept = accept_prob > rand_uni

        if torch.cuda.is_available():
            accept = accept.type(torch.FloatTensor).cuda()
        else:
            accept = accept.type(torch.FloatTensor)
        
        accept = accept.view(k, model.B, 1)

        z = (accept * z) + ((1-accept) * z0)

        #Adapt step size
        avg_acceptance_rate = torch.mean(accept)

        if avg_acceptance_rate.cpu().data.numpy() > .7:
            step_size = 1.02 * step_size
        else:
            step_size = .98 * step_size

        if step_size < 0.0001:
            step_size = 0.0001
        if step_size > 0.5:
            step_size = 0.5

        return z, step_size




    # n_intermediate_dists = 10
    n_HMC_steps = 5
    step_size = .1

    retain_graph = False
    volatile_ = False
    requires_grad = False

    time_ = time.time()

    logws = []
    data_index= 0
    for i in range(int(len(data_x)/ batch_size)):

        #AIS

        schedule = np.linspace(0.,1.,n_intermediate_dists)
        model.B = batch_size

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size

        

        if torch.cuda.is_available():
            batch = Variable(torch.from_numpy(batch), volatile=volatile_, requires_grad=requires_grad).cuda()
            zeros = Variable(torch.zeros(model.B, model.z_size), volatile=volatile_, requires_grad=requires_grad).cuda() # [B,Z]
            logw = Variable(torch.zeros(k, model.B), volatile=True, requires_grad=requires_grad).cuda()
            grad_outputs = torch.ones(k, model.B).cuda()
        else:
            batch = Variable(torch.from_numpy(batch))
            zeros = Variable(torch.zeros(model.B, model.z_size)) # [B,Z]
            logw = Variable(torch.zeros(k, model.B))
            grad_outputs = torch.ones(k, model.B)


        #Encode x
        mean, logvar = model.encode(batch) #[B,Z]
        #Init z
        z, logpz, logqz = model.sample(mean, logvar, k=k)  #[P,B,Z], [P,B], [P,B]

        for (t0, t1) in zip(schedule[:-1], schedule[1:]):


            #logw = logw + logpt-1(zt-1) - logpt(zt-1)
            log_intermediate_1 = intermediate_dist(t0, z, mean, logvar, zeros, batch)
            log_intermediate_2 = intermediate_dist(t1, z, mean, logvar, zeros, batch)
            logw += log_intermediate_2 - log_intermediate_1



            #HMC dynamics
            intermediate_dist_func = lambda aaa: intermediate_dist(t1, aaa, mean, logvar, zeros, batch)
            z0, v0, z, v = hmc(z, intermediate_dist_func)

            #MH step
            z, step_size = mh_step(z0, v0, z, v, step_size, intermediate_dist_func)

        #log sum exp
        max_ = torch.max(logw,0)[0] #[B]
        logw = torch.log(torch.mean(torch.exp(logw - max_), 0)) + max_ #[B]

        logws.append(torch.mean(logw.cpu()).data.numpy())


        if i%display==0:
            print (i,len(data_x)/ batch_size, np.mean(logws))

    mean_ = np.mean(logws)
    print(mean_, 'T:', time.time()-time_)
    return mean_
     


