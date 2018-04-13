


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



import sys
sys.path.insert(0, 'utils')
from utils import lognormal2 as lognormal
from utils import lognormal333









class Gaussian(nn.Module):

    def __init__(self, hyper_config): #, mean, logvar):
        #mean,logvar: [B,Z]
        super(Gaussian, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        

        # self.B = mean.size()[0]
        # # self.z_size = mean.size()[1]
        self.z_size = hyper_config['z_size']
        self.x_size = hyper_config['x_size']
        # # dfas

        # self.mean = mean
        # self.logvar = logvar


    def sample(self, mean, logvar, k):

        self.B = mean.size()[0]

        eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        return z, logqz



    def logprob(self, z, mean, logvar):

        # self.B = mean.size()[0]

        # eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        logqz = lognormal(z, mean, logvar) #[P,B]

        return logqz











class Flow(nn.Module):

    def __init__(self, hyper_config):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.hyper_config = hyper_config
        # self.B = mean.size()[0]
        self.z_size = hyper_config['z_size']
        self.x_size = hyper_config['x_size']

        self.act_func = hyper_config['act_func']
        

        count =1

        # f(vT|x,vT)
        # rv_arch = [[self.x_size+self.z_size,200],[200,200],[200,self.z_size*2]]
        rv_arch = [[self.z_size,50],[50,50],[50,self.z_size*2]]
        self.rv_weights = []
        for i in range(len(rv_arch)):
            layer = nn.Linear(rv_arch[i][0], rv_arch[i][1])
            self.rv_weights.append(layer)
            self.add_module(str(count), layer)
            count+=1


        n_flows = 2
        self.n_flows = n_flows
        h_s = 50

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)],
                                [nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)]
                                ])
        
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
    

        # # q(v0)
        # self.q_v = Gaussian(self.hyper_config, torch.zeros(self.B, self.z_size), torch.zeros(self.B, self.z_size))

        # # q(z0)
        # self.q_z = Gaussian(self.hyper_config, mean, logvar)

 


    def norm_flow(self, params, z, v):
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



    def sample(self, mean, logvar, k):

        self.B = mean.size()[0]
        gaus = Gaussian(self.hyper_config)

        # q(z0)
        z, logqz0 = gaus.sample(mean, logvar, k)

        # q(v0)
        zeros = Variable(torch.zeros(self.B, self.z_size)).cuda()
        v, logqv0 = gaus.sample(zeros, zeros, k)


        #[PB,Z]
        z = z.view(-1,self.z_size)
        v = v.view(-1,self.z_size)

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z, v, logdet = self.norm_flow(params,z,v)
            logdetsum += logdet

        logdetsum = logdetsum.view(k,self.B)

        #r(vT|x,zT)
        #r(vT|zT)  try that
        out = z #[PB,Z]
        # print (out.size())
        # fasda
        for i in range(len(self.rv_weights)-1):
            out = self.act_func(self.rv_weights[i](out))
        out = self.rv_weights[-1](out)
        # print (out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]
        # r_vt = Gaussian(self.hyper_config, mean, logvar)



        v = v.view(k, self.B, self.z_size)
        z = z.view(k, self.B, self.z_size)

        mean = mean.contiguous().view(k, self.B, self.z_size)
        logvar = logvar.contiguous().view(k, self.B, self.z_size)

        # print (mean.size()) #[PB,Z]
        # print (v.size())   #[P,B,Z]
        # print (self.B)
        # print (k)

        # logrvT = gaus.logprob(v, mean, logvar)
        logrvT = lognormal333(v, mean, logvar)

        # print (logqz0.size())
        # print (logqv0.size())
        # print (logdetsum.size())
        # print (logrvT.size())
        # fadsf




        logpz = logqz0+logqv0-logdetsum-logrvT

        return z, logpz







































class HNF(nn.Module):

    def __init__(self, hyper_config):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(HNF, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.hyper_config = hyper_config
        # self.B = mean.size()[0]
        self.z_size = hyper_config['z_size']
        self.x_size = hyper_config['x_size']

        self.act_func = hyper_config['act_func']
        

        count =1

        # f(vT|x,vT)
        # rv_arch = [[self.x_size+self.z_size,200],[200,200],[200,self.z_size*2]]
        rv_arch = [[self.z_size,50],[50,50],[50,self.z_size*2]]
        self.rv_weights = []
        for i in range(len(rv_arch)):
            layer = nn.Linear(rv_arch[i][0], rv_arch[i][1])
            self.rv_weights.append(layer)
            self.add_module(str(count), layer)
            count+=1


        n_flows = 2
        self.n_flows = n_flows
        h_s = 50

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)],
                                [nn.Linear(self.z_size, h_s), nn.Linear(h_s, self.z_size), nn.Linear(h_s, self.z_size)]
                                ])
        
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
    

        # # q(v0)
        # self.q_v = Gaussian(self.hyper_config, torch.zeros(self.B, self.z_size), torch.zeros(self.B, self.z_size))

        # # q(z0)
        # self.q_z = Gaussian(self.hyper_config, mean, logvar)




    def norm_flow(self, params, z, v, logposterior):


        h = F.tanh(params[0][0](z))
        mew_ = params[0][1](h)
        sig_ = F.sigmoid(params[0][2](h)) #[PB,Z]

        z_reshaped = z.view(self.P, self.B, self.z_size)
        gradients = torch.autograd.grad(outputs=logposterior(z_reshaped), inputs=z_reshaped,
                          grad_outputs=self.grad_outputs,
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.detach()
        gradients = gradients.view(-1,self.z_size)

        gradients = torch.clamp(torch.abs(gradients), max=1000)



        v = v*sig_ + mew_*gradients
        logdet = torch.sum(torch.log(sig_), 1)



        h = F.tanh(params[1][0](v))
        mew_ = params[1][1](h)
        sig_ = F.sigmoid(params[1][2](h)) #[PB,Z]
        # z = z*sig_ + mew_
        z = z*sig_ + mew_  #*v  #which one is better?? this is more like HVI
        logdet2 = torch.sum(torch.log(sig_), 1)


        
        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z, v, logdet



    def sample(self, mean, logvar, k, logposterior):

        self.P = k
        self.B = mean.size()[0]

        if torch.cuda.is_available():
            self.grad_outputs = torch.ones(k, self.B).cuda()
        else:
            self.grad_outputs = torch.ones(k, self.B)


        gaus = Gaussian(self.hyper_config)

        # q(z0)
        z, logqz0 = gaus.sample(mean, logvar, k)

        # q(v0)
        zeros = Variable(torch.zeros(self.B, self.z_size)).cuda()
        v, logqv0 = gaus.sample(zeros, zeros, k)


        #[PB,Z]
        z = z.view(-1,self.z_size)
        v = v.view(-1,self.z_size)

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z, v, logdet = self.norm_flow(params,z,v, logposterior)
            logdetsum += logdet

        logdetsum = logdetsum.view(k,self.B)

        #r(vT|x,zT)
        #r(vT|zT)  try that
        out = z #[PB,Z]
        # print (out.size())
        # fasda
        for i in range(len(self.rv_weights)-1):
            out = self.act_func(self.rv_weights[i](out))
        out = self.rv_weights[-1](out)
        # print (out)
        mean = out[:,:self.z_size]
        logvar = out[:,self.z_size:]
        # r_vt = Gaussian(self.hyper_config, mean, logvar)



        v = v.view(k, self.B, self.z_size)
        z = z.view(k, self.B, self.z_size)

        mean = mean.contiguous().view(k, self.B, self.z_size)
        logvar = logvar.contiguous().view(k, self.B, self.z_size)

        # print (mean.size()) #[PB,Z]
        # print (v.size())   #[P,B,Z]
        # print (self.B)
        # print (k)

        # logrvT = gaus.logprob(v, mean, logvar)
        logrvT = lognormal333(v, mean, logvar)

        # print (logqz0.size())
        # print (logqv0.size())
        # print (logdetsum.size())
        # print (logrvT.size())
        # fadsf




        logpz = logqz0+logqv0-logdetsum-logrvT

        return z, logpz



















#NO AUX VAF
class Flow1(nn.Module):

    def __init__(self, hyper_config):#, mean, logvar):
        #mean,logvar: [B,Z]
        super(Flow1, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.hyper_config = hyper_config
        # self.B = mean.size()[0]
        self.z_size = hyper_config['z_size']
        self.x_size = hyper_config['x_size']

        self.act_func = hyper_config['act_func']
        

        count =1


        n_flows = 2
        self.n_flows = n_flows
        h_s = 50

        self.z_half_size = int(self.z_size / 2)

        
        self.flow_params = []
        for i in range(n_flows):
            #first is for v, second is for z
            self.flow_params.append([
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)],
                                [nn.Linear(self.z_half_size, h_s), nn.Linear(h_s, self.z_half_size), nn.Linear(h_s, self.z_half_size)]
                                ])
        
        for i in range(n_flows):

            self.add_module(str(count), self.flow_params[i][0][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][0])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][1])
            count+=1
            self.add_module(str(count), self.flow_params[i][0][2])
            count+=1
            self.add_module(str(count), self.flow_params[i][1][2])
            count+=1
    



    def norm_flow(self, params, z1, z2):
        # print (z.size())
        h = F.tanh(params[0][0](z1))
        mew_ = params[0][1](h)
        # sig_ = F.sigmoid(params[0][2](h)+5.) #[PB,Z]
        sig_ = F.sigmoid(params[0][2](h)) #[PB,Z]

        z2 = z2*sig_ + mew_
        logdet = torch.sum(torch.log(sig_), 1)


        h = F.tanh(params[1][0](z2))
        mew_ = params[1][1](h)
        # sig_ = F.sigmoid(params[1][2](h)+5.) #[PB,Z]
        sig_ = F.sigmoid(params[1][2](h)) #[PB,Z]
        z1 = z1*sig_ + mew_
        logdet2 = torch.sum(torch.log(sig_), 1)



        #[PB]
        logdet = logdet + logdet2
        #[PB,Z], [PB]
        return z1, z2, logdet



    def sample(self, mean, logvar, k):

        self.B = mean.size()[0]
        gaus = Gaussian(self.hyper_config)

        # q(z0)
        z, logqz0 = gaus.sample(mean, logvar, k)

        #[PB,Z]
        z = z.view(-1,self.z_size)
        # v = v.view(-1,self.z_size)

        #Split z  [PB,Z/2]
        z1 = z.narrow(1, 0, self.z_half_size)
        z2 = z.narrow(1, self.z_half_size, self.z_half_size) 

        #Transform
        logdetsum = 0.
        for i in range(self.n_flows):

            params = self.flow_params[i]

            # z, v, logdet = self.norm_flow([self.flow_params[i]],z,v)
            z1, z2, logdet = self.norm_flow(params,z1,z2)
            logdetsum += logdet

        logdetsum = logdetsum.view(k,self.B)

        #Put z back together  [PB,Z]
        z = torch.cat([z1,z2],1)

        z = z.view(k, self.B, self.z_size)


        logpz = logqz0-logdetsum

        return z, logpz











