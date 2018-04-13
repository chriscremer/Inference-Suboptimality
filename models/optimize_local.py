






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




def optimize_local_gaussian(logposterior, model, x):

    # print_ = 0

    # B = x.shape[0]
    B = x.size()[0] #batch size
    # input to log posterior is z, [P,B,Z]
    # I think B will be 1 for now



        # self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

    mean = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    logvar = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    optimizer = optim.Adam([mean, logvar], lr=.001)
    # time_ = time.time()
    # n_data = len(train_x)
    # arr = np.array(range(n_data))

    P = 50


    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 999999):

        #Sample
        eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        logpx = logposterior(z)

        # print (logpx)
        # print (logqz)

        # fsda

        # data_index= 0
        # for i in range(int(n_data/batch_size)):
            # batch = train_x[data_index:data_index+batch_size]
            # data_index += batch_size

            # batch = Variable(torch.from_numpy(batch)).type(self.dtype)
        optimizer.zero_grad()

        # elbo, logpxz, logqz = self.forward(batch, k=k)

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

        # break


        # if epoch%display_epoch==0:
        #     print ('Train Epoch: {}/{}'.format(epoch, epochs),
        #         'LL:{:.3f}'.format(-loss.data[0]),
        #         'logpxz:{:.3f}'.format(logpxz.data[0]),
        #         # 'logpz:{:.3f}'.format(logpz.data[0]),
        #         'logqz:{:.3f}'.format(logqz.data[0]),
        #         'T:{:.2f}'.format(time.time()-time_),
        #         )

        #     time_ = time.time()


    # Compute VAE and IWAE bounds



    #Sample
    eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
    z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
    logqz = lognormal(z, mean, logvar) #[P,B]

    # print (logqz)
    # fad
    logpx = logposterior(z)

    elbo = logpx-logqz #[P,B]
    vae = torch.mean(elbo)

    max_ = torch.max(elbo, 0)[0] #[B]
    elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    iwae = torch.mean(elbo_)

    return vae, iwae








def optimize_local_gaussian_mean_logvar(logposterior, model, x):

    # B = x.shape[0]
    B = x.size()[0] #batch size
    # input to log posterior is z, [P,B,Z]
    # I think B will be 1 for now



        # self.zeros = Variable(torch.zeros(self.B, self.z_size).type(self.dtype))

    mean = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    logvar = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    optimizer = optim.Adam([mean, logvar], lr=.001)
    # time_ = time.time()
    # n_data = len(train_x)
    # arr = np.array(range(n_data))

    P = 50


    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 999999):

        if quick:
        # if 1:

            break

        #Sample
        eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        logpx = logposterior(z)

        # print (logpx)
        # print (logqz)

        # fsda

        # data_index= 0
        # for i in range(int(n_data/batch_size)):
            # batch = train_x[data_index:data_index+batch_size]
            # data_index += batch_size

            # batch = Variable(torch.from_numpy(batch)).type(self.dtype)
        optimizer.zero_grad()

        # elbo, logpxz, logqz = self.forward(batch, k=k)

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

            # print (epoch, last_100_avg, consecutive_worse,mean)
            # print (torch.mean(logpx))

            last_100 = []


        # if epoch%display_epoch==0:
        #     print ('Train Epoch: {}/{}'.format(epoch, epochs),
        #         'LL:{:.3f}'.format(-loss.data[0]),
        #         'logpxz:{:.3f}'.format(logpxz.data[0]),
        #         # 'logpz:{:.3f}'.format(logpz.data[0]),
        #         'logqz:{:.3f}'.format(logqz.data[0]),
        #         'T:{:.2f}'.format(time.time()-time_),
        #         )

        #     time_ = time.time()


    # Compute VAE and IWAE bounds



    #Sample
    # eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
    # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
    # logqz = lognormal(z, mean, logvar) #[P,B]

    # # print (logqz)
    # # fad
    # logpx = logposterior(z)

    # elbo = logpx-logqz #[P,B]
    # vae = torch.mean(elbo)

    # max_ = torch.max(elbo, 0)[0] #[B]
    # elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    # iwae = torch.mean(elbo_)

    return mean, logvar







































def optimize_local_expressive(logposterior, model, x):


 
    def norm_flow(params, z, v):
        # print (z.size())
        h = F.tanh(params[0][0](z))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = F.sigmoid(params[0][2](h)) #[PB,Z]

        v = v*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)
        h = F.tanh(params[1][0](v))
        mew_ = params[1][1](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = F.sigmoid(params[1][2](h)) #[PB,Z]
        z = z*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z, v, logdet



    def sample(k):

        P = k

        # #Sample
        # eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        # logqz = lognormal(z, mean, logvar) #[P,B]

        # logpx = logposterior(z)
        # optimizer.zero_grad()


        #q(v|x)
        # out = x
        # for i in range(len(self.qv_weights)-1):
        #     out = self.act_func(self.qv_weights[i](out))
        # out = self.qv_weights[-1](out)
        # mean = out[:,:self.z_size]
        # logvar = out[:,self.z_size:]

        #Sample v0
        eps = Variable(torch.FloatTensor(k, B, z_size).normal_().type(model.dtype)) #[P,B,Z]
        # print (eps)
        v = eps.mul(torch.exp(.5*logvar_v)) + mean_v  #[P,B,Z]
        logqv0 = lognormal(v, mean_v, logvar_v) #[P,B]

        #[PB,Z]
        v = v.view(-1,model.z_size)

        # print ('v', v)
        # print (v)
        # fsaf

        # print(v)
        # fasd
        #[PB,X]
        # x_tiled = x.repeat(k,1)
        #[PB,X+Z]
        # xv = torch.cat((x_tiled, v),1)

        #q(z|x,v)
        out = v
        for i in range(len(qz_weights)-1):
            out = act_func(qz_weights[i](out))
        out = qz_weights[-1](out)
        mean = out[:,:z_size]
        logvar = out[:,z_size:] #+5.

        # print (mean)

        # B = x.size()[0]
        # print (self.B, 'B')
        #Sample z0
        eps = Variable(torch.FloatTensor(k, B, z_size).normal_().type(model.dtype)) #[P,B,Z]

        # print (mean)
        mean = mean.contiguous().view(P,B,model.z_size)
        logvar = logvar.contiguous().view(P,B,model.z_size)

        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        # print (z.size(),'z')

        # mean = mean.contiguous().view(P*B,model.z_size)
        # logvar = logvar.contiguous().view(P*B,model.z_size)
        # print ('z', z)


        # print (z)
        # fad

        logqz0 = lognormal333(z, mean, logvar) #[P,B]

        #[PB,Z]
        z = z.view(-1,z_size)

        logdetsum = 0.
        for i in range(n_flows):

            z, v, logdet = norm_flow(params[i],z,v)
            logdetsum += logdet


        # xz = torch.cat((x_tiled,z),1)

        #r(vT|x,zT)
        out = z
        for i in range(len(rv_weights)-1):
            out = act_func(rv_weights[i](out))
        out = rv_weights[-1](out)
        mean = out[:,:model.z_size]
        logvar = out[:,model.z_size:]

        mean = mean.contiguous().view(P,B,model.z_size)
        logvar = logvar.contiguous().view(P,B,model.z_size)

        v = v.view(k,B,model.z_size)
        logrvT = lognormal333(v, mean, logvar) #[P,B]

        z = z.view(k,B,model.z_size)
        # print ('z2', z)

        logdetsum = logdetsum.view(-1, 1)

        # print ('logqz0',logqz0)
        # print ('logqv0',logqv0)
        # print ('logdetsum',logdetsum)
        # print ('logrvT',logrvT)

        logq = logqz0+logqv0-logdetsum-logrvT

        # print ('logq', logq)
        # fafd

        # print (torch.mean(logqz0),torch.mean(logqv0),torch.mean(logdetsum),torch.mean(logrvT))

        return z, logq







    x_size = 784
    z_size = 50

    # hyper_config = model.hyper_config
    hyper_config = { 
                    'x_size': x_size,
                    'z_size': z_size,
                    'act_func': F.tanh,# F.relu,
                    'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                    'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                    # 'q_dist': hnf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
                    'n_flows': 2,
                    # 'qv_arch': [[x_size,200],[200,200],[200,z_size*2]],
                    'qz_arch': [[z_size,200],[200,200],[200,z_size*2]],
                    'rv_arch': [[z_size,200],[200,200],[200,z_size*2]],
                    'flow_hidden_size': 100
                }



    # B = x.shape[0]
    B = x.size()[0] #batch size

    n_flows = 2 #hyper_config['n_flows']

    z_size = model.z_size
    x_size = model.x_size
    act_func = model.act_func

    all_params = []

    # #q(v|x)
    # self.qv_weights = []
    # for i in range(len(hyper_config['qv_arch'])):
    #     self.qv_weights.append(nn.Linear(hyper_config['qv_arch'][i][0], hyper_config['qv_arch'][i][1]))
    mean_v = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    logvar_v = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    all_params.append(mean_v)
    all_params.append(logvar_v)


    #q(z|x,v)
    qz_weights = []
    for i in range(len(hyper_config['qz_arch'])):
        if torch.cuda.is_available():
            qz_weights.append(nn.Linear(hyper_config['qz_arch'][i][0], hyper_config['qz_arch'][i][1]).cuda())
        else:
            qz_weights.append(nn.Linear(hyper_config['qz_arch'][i][0], hyper_config['qz_arch'][i][1]))

        all_params.append(qz_weights[i].weight)
    #r(v|x,z)
    rv_weights = []
    for i in range(len(hyper_config['rv_arch'])):
        if torch.cuda.is_available():
            rv_weights.append(nn.Linear(hyper_config['rv_arch'][i][0], hyper_config['rv_arch'][i][1]).cuda())
        else:
            rv_weights.append(nn.Linear(hyper_config['rv_arch'][i][0], hyper_config['rv_arch'][i][1]))

        all_params.append(rv_weights[i].weight)

    h_s = hyper_config['flow_hidden_size']

    params = []
    for i in range(n_flows):

        #first is for v, second is for z
        if torch.cuda.is_available():
            aaa = [
                    [nn.Linear(model.z_size, h_s).cuda(), nn.Linear(h_s, model.z_size).cuda(), nn.Linear(h_s, model.z_size).cuda()],
                    [nn.Linear(model.z_size, h_s).cuda(), nn.Linear(h_s, model.z_size).cuda(), nn.Linear(h_s, model.z_size).cuda()]
                    ]
        else:
            aaa = [
                    [nn.Linear(model.z_size, h_s), nn.Linear(h_s, model.z_size), nn.Linear(h_s, model.z_size)],
                    [nn.Linear(model.z_size, h_s), nn.Linear(h_s, model.z_size), nn.Linear(h_s, model.z_size)]
                    ]


        params.append(aaa)

        all_params.append(aaa[0][0].weight)
        all_params.append(aaa[0][1].weight)
        all_params.append(aaa[0][2].weight)
        all_params.append(aaa[1][0].weight)
        all_params.append(aaa[1][1].weight)
        all_params.append(aaa[1][2].weight)



    optimizer = optim.Adam(all_params, lr=.001)
    # time_ = time.time()
    # n_data = len(train_x)
    # arr = np.array(range(n_data))





    P = 50
    k = P


    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 999999):


        z, logq = sample(k)


        logpx = logposterior(z)


        optimizer.zero_grad()
        loss = -(torch.mean(logpx-logq))
        loss_np = loss.data.cpu().numpy()
        # print ()


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

            print (epoch, last_100_avg, consecutive_worse)
            # print (torch.mean(logpx).data.cpu().numpy())
            # print (torch.mean(logqz0).data.cpu().numpy(),torch.mean(logqv0).data.cpu().numpy(),torch.mean(logdetsum).data.cpu().numpy(),torch.mean(logrvT).data.cpu().numpy())

            last_100 = []





    #Sample
    k=1000
    # eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
    # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
    # logqz = lognormal(z, mean, logvar) #[P,B]
    z, logq = sample(k)
    logpx = logposterior(z)

    # print (logq)
    # print (logpx)

    elbo = logpx-logq #[P,B]

    # print (elbo)
    vae = torch.mean(elbo)

    # print (vae)

    max_ = torch.max(elbo, 0)[0] #[B]
    elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    iwae = torch.mean(elbo_)

    # print (iwae)
    # fada

    return vae, iwae






























def optimize_local_expressive_only_sample(logposterior, model, x):


 
    def norm_flow(params, z, v):
        # print (z.size())
        h = F.tanh(params[0][0](z))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = F.sigmoid(params[0][2](h)) #[PB,Z]
        v = v*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)
        h = F.tanh(params[1][0](v))
        mew_ = params[1][1](h)
        sig_ = F.sigmoid(params[1][2](h)) #[PB,Z]
        z = z*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z, v, logdet



    def sample(k):

        P = k

        # #Sample
        # eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        # logqz = lognormal(z, mean, logvar) #[P,B]

        # logpx = logposterior(z)
        # optimizer.zero_grad()


        #q(v|x)
        # out = x
        # for i in range(len(self.qv_weights)-1):
        #     out = self.act_func(self.qv_weights[i](out))
        # out = self.qv_weights[-1](out)
        # mean = out[:,:self.z_size]
        # logvar = out[:,self.z_size:]

        #Sample v0
        eps = Variable(torch.FloatTensor(k, B, z_size).normal_().type(model.dtype)) #[P,B,Z]
        # print (eps)
        v = eps.mul(torch.exp(.5*logvar_v)) + mean_v  #[P,B,Z]
        logqv0 = lognormal(v, mean_v, logvar_v) #[P,B]

        #[PB,Z]
        v = v.view(-1,model.z_size)
        # print (v)
        # fsaf

        # print(v)
        # fasd
        #[PB,X]
        # x_tiled = x.repeat(k,1)
        #[PB,X+Z]
        # xv = torch.cat((x_tiled, v),1)

        #q(z|x,v)
        out = v
        for i in range(len(qz_weights)-1):
            out = act_func(qz_weights[i](out))
        out = qz_weights[-1](out)
        mean = out[:,:z_size]
        logvar = out[:,z_size:] + 5.

        # print (mean)

        # B = x.size()[0]
        # print (self.B, 'B')
        #Sample z0
        eps = Variable(torch.FloatTensor(k, B, z_size).normal_().type(model.dtype)) #[P,B,Z]

        # print (mean)
        mean = mean.contiguous().view(P,B,model.z_size)
        logvar = logvar.contiguous().view(P,B,model.z_size)

        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        # print (z.size(),'z')

        # mean = mean.contiguous().view(P*B,model.z_size)
        # logvar = logvar.contiguous().view(P*B,model.z_size)



        # print (z)
        # fad

        logqz0 = lognormal333(z, mean, logvar) #[P,B]

        #[PB,Z]
        z = z.view(-1,z_size)

        logdetsum = 0.
        for i in range(n_flows):

            z, v, logdet = norm_flow(params[i],z,v)
            logdetsum += logdet


        # xz = torch.cat((x_tiled,z),1)

        #r(vT|x,zT)
        out = z
        for i in range(len(rv_weights)-1):
            out = act_func(rv_weights[i](out))
        out = rv_weights[-1](out)
        mean = out[:,:model.z_size]
        logvar = out[:,model.z_size:]

        mean = mean.contiguous().view(P,B,model.z_size)
        logvar = logvar.contiguous().view(P,B,model.z_size)

        v = v.view(k,B,model.z_size)
        logrvT = lognormal333(v, mean, logvar) #[P,B]

        z = z.view(k,B,model.z_size)


        logq = logqz0+logqv0-logdetsum-logrvT

        # print (torch.mean(logqz0),torch.mean(logqv0),torch.mean(logdetsum),torch.mean(logrvT))

        return z, logq







    x_size = 784
    z_size = 2

    l_size = 100

    # hyper_config = model.hyper_config
    hyper_config = { 
                    'x_size': x_size,
                    'z_size': z_size,
                    'act_func': F.tanh,# F.relu,
                    'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                    'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                    # 'q_dist': hnf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
                    'n_flows': 2,
                    # 'qv_arch': [[x_size,200],[200,200],[200,z_size*2]],
                    'qz_arch': [[z_size,l_size],[l_size,l_size],[l_size,z_size*2]],
                    'rv_arch': [[z_size,l_size],[l_size,l_size],[l_size,z_size*2]],
                    'flow_hidden_size': 30
                }



    # B = x.shape[0]
    B = x.size()[0] #batch size

    n_flows = 2 #hyper_config['n_flows']

    z_size = model.z_size
    x_size = model.x_size
    act_func = model.act_func

    all_params = []

    # #q(v|x)
    # self.qv_weights = []
    # for i in range(len(hyper_config['qv_arch'])):
    #     self.qv_weights.append(nn.Linear(hyper_config['qv_arch'][i][0], hyper_config['qv_arch'][i][1]))
    mean_v = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    logvar_v = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    all_params.append(mean_v)
    all_params.append(logvar_v)


    #q(z|x,v)
    qz_weights = []
    for i in range(len(hyper_config['qz_arch'])):
        qz_weights.append(nn.Linear(hyper_config['qz_arch'][i][0], hyper_config['qz_arch'][i][1]).cuda())
        all_params.append(qz_weights[i].weight)
    #r(v|x,z)
    rv_weights = []
    for i in range(len(hyper_config['rv_arch'])):
        rv_weights.append(nn.Linear(hyper_config['rv_arch'][i][0], hyper_config['rv_arch'][i][1]).cuda())
        all_params.append(rv_weights[i].weight)

    h_s = hyper_config['flow_hidden_size']

    params = []
    for i in range(n_flows):

        #first is for v, second is for z
        aaa = [
                [nn.Linear(model.z_size, h_s).cuda(), nn.Linear(h_s, model.z_size).cuda(), nn.Linear(h_s, model.z_size).cuda()],
                [nn.Linear(model.z_size, h_s).cuda(), nn.Linear(h_s, model.z_size).cuda(), nn.Linear(h_s, model.z_size).cuda()]
                ]
        params.append(aaa)

        all_params.append(aaa[0][0].weight)
        all_params.append(aaa[0][1].weight)
        all_params.append(aaa[0][2].weight)
        all_params.append(aaa[1][0].weight)
        all_params.append(aaa[1][1].weight)
        all_params.append(aaa[1][2].weight)



    optimizer = optim.Adam(all_params, lr=.001)
    # time_ = time.time()
    # n_data = len(train_x)
    # arr = np.array(range(n_data))



    # if load_from == '':
    if 1:


        P = 50
        k = P


        last_100 = []
        best_last_100_avg = -1
        consecutive_worse = 0
        for epoch in range(1, 999999):

            if quick:
                break


            z, logq = sample(k)


            logpx = logposterior(z)


            optimizer.zero_grad()
            loss = -(torch.mean(logpx-logq))
            loss_np = loss.data.cpu().numpy()
            # print ()


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

                # print (epoch, last_100_avg, consecutive_worse)
                # print(z[0])
                # print (torch.mean(logpx).data.cpu().numpy())
                # print (torch.mean(logqz0).data.cpu().numpy(),torch.mean(logqv0).data.cpu().numpy(),torch.mean(logdetsum).data.cpu().numpy(),torch.mean(logrvT).data.cpu().numpy())


                last_100 = []

        # if save_to != '':
        #     pickle.dump(all_params, open(save_to, "wb" ))
        #     print ('saved', save_to)




    else:

        #load
        loaded_params = pickle.load(open(load_from, "rb" ))
        print ('loaded', load_from)

        # for i in range(len(all_params)):
        #     print (loaded_params[i])
        #     # all_params[i] = loaded_params[i]


        count =0

        # mean_v = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
        # logvar_v = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
        # all_params.append(mean_v)
        # all_params.append(logvar_v) 

        mean_v = loaded_params[count]
        count+=1
        logvar_v = loaded_params[count]
        count+=1




        #q(z|x,v)
        qz_weights = []
        for i in range(len(hyper_config['qz_arch'])):
            # qz_weights.append(nn.Linear(hyper_config['qz_arch'][i][0], hyper_config['qz_arch'][i][1]).cuda())
            # all_params.append(qz_weights[i].weight)

            l1 = nn.Linear(hyper_config['qz_arch'][i][0], hyper_config['qz_arch'][i][1]).cuda()
            l1.weight = loaded_params[count]

            # qz_weights.append(loaded_params[count])
            qz_weights.append(l1)

            count+=1

            

        #r(v|x,z)
        rv_weights = []
        for i in range(len(hyper_config['rv_arch'])):
            # rv_weights.append(nn.Linear(hyper_config['rv_arch'][i][0], hyper_config['rv_arch'][i][1]).cuda())
            # all_params.append(rv_weights[i].weight)

            l1 = nn.Linear(hyper_config['rv_arch'][i][0], hyper_config['rv_arch'][i][1]).cuda()
            l1.weight = loaded_params[count]

            # rv_weights.append(loaded_params[count])
            rv_weights.append(l1)

            count+=1

        h_s = hyper_config['flow_hidden_size']

        params = []
        for i in range(n_flows):

            # #first is for v, second is for z
            # aaa = [
            #         [nn.Linear(model.z_size, h_s).cuda(), nn.Linear(h_s, model.z_size).cuda(), nn.Linear(h_s, model.z_size).cuda()],
            #         [nn.Linear(model.z_size, h_s).cuda(), nn.Linear(h_s, model.z_size).cuda(), nn.Linear(h_s, model.z_size).cuda()]
            #         ]
            # params.append(aaa)

            # all_params.append(aaa[0][0].weight)
            # all_params.append(aaa[0][1].weight)
            # all_params.append(aaa[0][2].weight)
            # all_params.append(aaa[1][0].weight)
            # all_params.append(aaa[1][1].weight)
            # all_params.append(aaa[1][2].weight)


            l1 = nn.Linear(model.z_size, h_s).cuda()
            l1.weight = loaded_params[count]

            l2 = nn.Linear(h_s, model.z_size).cuda()
            l2.weight = loaded_params[count+1]

            l3 = nn.Linear(h_s, model.z_size).cuda()
            l3.weight = loaded_params[count+2]   

            l4 = nn.Linear(model.z_size, h_s).cuda()
            l4.weight = loaded_params[count+3]

            l5 = nn.Linear(h_s, model.z_size).cuda()
            l5.weight = loaded_params[count+4]

            l6 = nn.Linear(h_s, model.z_size).cuda()
            l6.weight = loaded_params[count+5]        



            # aaa = [
            #         [loaded_params[count], loaded_params[count+1], loaded_params[count+2]],
            #         [loaded_params[count+3], loaded_params[count+4], loaded_params[count+5]]
            #         ]
            aaa = [
                    [l1, l2, l3],
                    [l4, l5, l5]
                    ]
                            
            count+=6

            params.append(aaa)









    # fasd

    #Sample
    k=1000
    # eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
    # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
    # logqz = lognormal(z, mean, logvar) #[P,B]
    z, logq = sample(k)
    # logpx = logposterior(z)

    # elbo = logpx-logq #[P,B]
    # vae = torch.mean(elbo)

    # max_ = torch.max(elbo, 0)[0] #[B]
    # elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
    # iwae = torch.mean(elbo_)

    return z

































# class aux_nf__(nn.Module):

#     def __init__(self, model, hyper_config):
#         super(aux_nf__, self).__init__()


#         if torch.cuda.is_available():
#             self.dtype = torch.cuda.FloatTensor
#         else:
#             self.dtype = torch.FloatTensor



#         self.z_size = model.z_size
#         z_size = self.z_size
#         self.x_size = model.x_size
#         x_size = self.x_size
#         self.act_func = model.act_func



#         hyper_config = { 
#                         'x_size': x_size,
#                         'z_size': z_size,
#                         'act_func': F.tanh,# F.relu,
#                         'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                         'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                         # 'q_dist': aux_nf, #hnf,#aux_nf,#flow1,#standard,#, #, #, #,#, #,# ,
#                         'n_flows': 2,
#                         # 'qv_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                         'qz_arch': [[z_size,200],[200,200],[200,z_size*2]],
#                         'rv_arch': [[z_size,200],[200,200],[200,z_size*2]],
#                         'flow_hidden_size': 100
#                     }

#         self.n_flows = hyper_config['n_flows']


#         #q(v|x)
#         self.qv_weights = []
#         # for i in range(len(hyper_config['qv_arch'])):
#         #     self.qv_weights.append(nn.Linear(hyper_config['qv_arch'][i][0], hyper_config['qv_arch'][i][1]))
#         self.qv_weights.append(Variable(torch.zeros(1, model.z_size).type(model.dtype), requires_grad=True))
#         self.qv_weights.append(Variable(torch.zeros(1, model.z_size).type(model.dtype), requires_grad=True))

#         #q(z|x,v)
#         self.qz_weights = []
#         for i in range(len(hyper_config['qz_arch'])):
#             self.qz_weights.append(nn.Linear(hyper_config['qz_arch'][i][0], hyper_config['qz_arch'][i][1]))
#         #r(v|x,z)
#         self.rv_weights = []
#         for i in range(len(hyper_config['rv_arch'])):
#             self.rv_weights.append(nn.Linear(hyper_config['rv_arch'][i][0], hyper_config['rv_arch'][i][1]))

#         h_s = hyper_config['flow_hidden_size']

#         self.params = []
#         for i in range(self.n_flows):

#             #first is for v, second is for z
#             self.params.append([
#                                 [nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)],
#                                 [nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)]
#                                 ])

#         # self.param_list = nn.ParameterList(self.params)


#         count =1
#         # for i in range(len(self.qv_weights)):
#         #     self.add_module(str(count), self.qv_weights[i])
#         #     count+=1
#         for i in range(len(self.qz_weights)):
#             self.add_module(str(count), self.qz_weights[i])
#             count+=1
#         for i in range(len(self.rv_weights)):
#             self.add_module(str(count), self.rv_weights[i])
#             count+=1

#         # count =1
#         for i in range(self.n_flows):

#             self.add_module(str(count), self.params[i][0][0])
#             count+=1
#             self.add_module(str(count), self.params[i][1][0])
#             count+=1
#             self.add_module(str(count), self.params[i][0][1])
#             count+=1
#             self.add_module(str(count), self.params[i][1][1])
#             count+=1
#             self.add_module(str(count), self.params[i][0][2])
#             count+=1
#             self.add_module(str(count), self.params[i][1][2])
#             count+=1
    

#     def forward(self, k):
#         '''
#         k: number of samples
#         x: [B,X]
#         logposterior(z) -> [P,B]
#         '''

#         # self.B = x.size()[0]
#         self.P = k


#         #q(v|x)
#         # out = x
#         # for i in range(len(self.qv_weights)-1):
#         #     out = self.act_func(self.qv_weights[i](out))
#         # out = self.qv_weights[-1](out)
#         # mean = out[:,:self.z_size]
#         # logvar = out[:,self.z_size:]

#         mean = self.qv_weights[0]
#         logvar = self.qv_weights[1]


#         #Sample v0
#         eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
#         v = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         logqv0 = lognormal(v, mean, logvar) #[P,B]

#         #[PB,Z]
#         v = v.view(-1,self.z_size)
#         #[PB,X]
#         # x_tiled = x.repeat(k,1)
#         #[PB,X+Z]
#         # print (x_tiled.size())
#         # print (v.size())

#         # xv = torch.cat((x_tiled, v),1)

#         #q(z|x,v)
#         out = v
#         for i in range(len(self.qz_weights)-1):
#             out = self.act_func(self.qz_weights[i](out))
#         out = self.qz_weights[-1](out)
#         mean = out[:,:self.z_size]
#         logvar = out[:,self.z_size:]

#         # self.B = x.size()[0]
#         # print (self.B, 'B')
#         #Sample z0
#         eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
#         # print (eps.size(),'eps')
#         # print (mean.size(),'mean')
#         # print (self.P, 'P')

#         # print (mean)
#         mean = mean.contiguous().view(self.P,self.B,self.z_size)
#         logvar = logvar.contiguous().view(self.P,self.B,self.z_size)

#         z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         # print (z.size(),'z')

#         # mean = mean.contiguous().view(self.P*self.B,self.z_size)
#         # logvar = logvar.contiguous().view(self.P*self.B,self.z_size)

#         logqz0 = lognormal(z, mean, logvar) #[P,B]

#         #[PB,Z]
#         z = z.view(-1,self.z_size)

#         # print (z.size())

#         # print (z)
#         # print (v)
#         # fasdf

#         logdetsum = 0.
#         for i in range(self.n_flows):

#             z, v, logdet = self.norm_flow(self.params[i],z,v)
#             logdetsum += logdet


#         # xz = torch.cat((x_tiled,z),1)

#         #r(vT|x,zT)
#         out = z
#         for i in range(len(self.rv_weights)-1):
#             out = self.act_func(self.rv_weights[i](out))
#         out = self.rv_weights[-1](out)
#         mean = out[:,:self.z_size]
#         logvar = out[:,self.z_size:]

#         v = v.view(k,self.B,self.z_size)
#         # print (v)
#         # fdas
#         logrvT = lognormal(v, mean, logvar) #[P,B]

#         z = z.view(k,self.B,self.z_size)

#         print (torch.mean(logqz0),torch.mean(logqv0),torch.mean(logdetsum),torch.mean(logrvT))

#         return z, logqz0+logqv0-logdetsum-logrvT



 
#     def norm_flow(self, params, z, v):

#         # print (z.size())
#         h = F.tanh(params[0][0](z))
#         mew_ = params[0][1](h)
#         sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]


#         v = v*sig_ + mew_

#         logdet = torch.sum(torch.log(sig_), 1)


#         h = F.tanh(params[1][0](v))
#         mew_ = params[1][1](h)
#         sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]

#         z = z*sig_ + mew_

#         logdet2 = torch.sum(torch.log(sig_), 1)

#         #[PB]
#         logdet = logdet + logdet2
        
#         #[PB,Z], [PB]
#         return z, v, logdet




#     def train_and_eval(self, logposterior, model, x):

#         self.B = x.size()[0]

#         optimizer = optim.Adam(self.parameters(), lr=.001)
#         # time_ = time.time()
#         # n_data = len(train_x)
#         # arr = np.array(range(n_data))

#         P = 50


#         last_100 = []
#         best_last_100_avg = -1
#         consecutive_worse = 0
#         for epoch in range(1, 999999):

#             # print (epoch)

#             # #Sample
#             # eps = Variable(torch.FloatTensor(P, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
#             # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#             # logqz = lognormal(z, mean, logvar) #[P,B]

#             z, logq = self.forward(P)

#             logpx = logposterior(z)

#             optimizer.zero_grad()

#             loss = -(torch.mean(logpx-logq))

#             loss_np = loss.data.cpu().numpy()
#             print (epoch, loss_np, torch.mean(logpx))
#             # fasfaf

#             loss.backward()
#             optimizer.step()

#             last_100.append(loss_np)
#             if epoch % 100 ==0:

#                 last_100_avg = np.mean(last_100)
#                 if last_100_avg< best_last_100_avg or best_last_100_avg == -1:
#                     consecutive_worse=0
#                     best_last_100_avg = last_100_avg
#                 else:
#                     consecutive_worse +=1 
#                     # print(consecutive_worse)
#                     if consecutive_worse> 10:
#                         # print ('done')
#                         break

#                 # print (epoch, last_100_avg, consecutive_worse)

#                 last_100 = []



#         #Sample
#         eps = Variable(torch.FloatTensor(1000, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
#         z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
#         logqz = lognormal(z, mean, logvar) #[P,B]
#         logpx = logposterior(z)

#         elbo = logpx-logqz #[P,B]
#         vae = torch.mean(elbo)

#         max_ = torch.max(elbo, 0)[0] #[B]
#         elbo_ = torch.log(torch.mean(torch.exp(elbo - max_), 0)) + max_ #[B]
#         iwae = torch.mean(elbo_)

#         return vae, iwae



















































def optimize_local_expressive_only_sample_2(logposterior, model, x):

    # no AUX


    def norm_flow(params, z):

        # [Z]
        mask = Variable(torch.zeros(model.z_size)).type(model.dtype)
        mask[:int(model.z_size/2)] = 1.
        mask = mask.view(1,1,-1)

        # [P,B,Z]
        z1 = z*mask
        # [PB,Z]
        z1 = z1.view(-1, model.z_size)

        h = F.tanh(params[0](z1))
        mew_ = params[1](h)
        sig_ = F.sigmoid(params[2](h))+1.#) #[PB,Z]

        z = z.view(-1, model.z_size)
        mask = mask.view(1, -1)

        z2 = (z*sig_ +mew_)*(1-mask)
        z = z1 + z2
        # [PB]
        logdet = torch.sum((1-mask)*torch.log(sig_), 1)
        # [P,B]
        logdet = logdet.view(-1,B)
        #[P,B,Z]
        z = z.view(-1,B,model.z_size)


        #Other half

        # [Z]
        mask2 = Variable(torch.zeros(model.z_size)).type(model.dtype)
        mask2[int(model.z_size/2):] = 1.
        mask = mask2.view(1,1,-1)

        # [P,B,Z]
        z1 = z*mask
        # [PB,Z]
        z1 = z1.view(-1, model.z_size)

        h = F.tanh(params[0](z1))
        mew_ = params[1](h)
        sig_ = F.sigmoid(params[2](h))+1.#) #[PB,Z]

        z = z.view(-1, model.z_size)
        mask = mask.view(1, -1)

        z2 = (z*sig_ +mew_)*(1-mask)
        z = z1 + z2
        # [PB]
        logdet2 = torch.sum((1-mask)*torch.log(sig_), 1)
        # [P,B]
        logdet2 = logdet2.view(-1,B)
        #[P,B,Z]
        z = z.view(-1,B,model.z_size)

        logdet = logdet + logdet2
        

        return z, logdet





    def sample(k):

        P = k

        #Sample
        eps = Variable(torch.FloatTensor(k, B, model.z_size).normal_().type(model.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        logdetsum = 0.
        for i in range(n_flows):

            z, logdet = norm_flow(params[i],z)
            logdetsum += logdet

        logq = logqz - logdetsum
        return z, logq



    x_size = 784
    z_size = 2

    hyper_config = { 
                    'x_size': x_size,
                    'z_size': z_size,
                    'act_func': F.tanh,# F.relu,
                    'n_flows': 2,
                    'flow_hidden_size': 30
                }



    # B = x.shape[0]
    B = x.size()[0] #batch size

    n_flows = 2 #hyper_config['n_flows']

    z_size = model.z_size
    x_size = model.x_size
    act_func = model.act_func

    all_params = []


    mean = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)
    logvar = Variable(torch.zeros(B, model.z_size).type(model.dtype), requires_grad=True)

    all_params.append(mean)
    all_params.append(logvar)


    h_s = hyper_config['flow_hidden_size']

    params = []
    for i in range(n_flows):
        params.append([nn.Linear(z_size, h_s).cuda(), nn.Linear(h_s, z_size).cuda(), nn.Linear(h_s, z_size).cuda()])

        all_params.append(params[i][0].weight)
        all_params.append(params[i][1].weight)
        all_params.append(params[i][2].weight)




    optimizer = optim.Adam(all_params, lr=.01)


    P = 50
    k = P


    last_100 = []
    best_last_100_avg = -1
    consecutive_worse = 0
    for epoch in range(1, 999999):

        if quick:
            break


        z, logq = sample(k)


        logpx = logposterior(z)


        optimizer.zero_grad()
        loss = -(torch.mean(logpx-logq))
        loss_np = loss.data.cpu().numpy()
        # print ()


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

            print (epoch, last_100_avg, consecutive_worse)
            # print(z[0])
            # print (torch.mean(logpx).data.cpu().numpy())
            # print (torch.mean(logqz0).data.cpu().numpy(),torch.mean(logqv0).data.cpu().numpy(),torch.mean(logdetsum).data.cpu().numpy(),torch.mean(logrvT).data.cpu().numpy())


            last_100 = []


    #Sample
    k=1000

    z, logq = sample(k)


    return z














