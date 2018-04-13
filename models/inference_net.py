




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



from distributions import Gaussian
from distributions import Flow







class standard(nn.Module):

    def __init__(self, hyper_config):
        super(standard, self).__init__()

        if torch.cuda.is_available():
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.hyper_config = hyper_config

        self.z_size = hyper_config['z_size']
        self.x_size = hyper_config['x_size']
        self.act_func = hyper_config['act_func']


        #Encoder
        self.encoder_weights = []
        self.layer_norms = []
        for i in range(len(hyper_config['encoder_arch'])):
            self.encoder_weights.append(nn.Linear(hyper_config['encoder_arch'][i][0], hyper_config['encoder_arch'][i][1]))
            
            # if i != len(hyper_config['encoder_arch'])-1:
            #     self.layer_norms.append(LayerNorm(hyper_config['encoder_arch'][i][1]))

        count =1
        for i in range(len(self.encoder_weights)):
            self.add_module(str(count), self.encoder_weights[i])
            count+=1

            # if i != len(hyper_config['encoder_arch'])-1:
            #     self.add_module(str(count), self.layer_norms[i])
            #     count+=1         



        # self.q = Gaussian(self.hyper_config) #, mean, logvar)
        # self.q = Flow(self.hyper_config)#, mean, logvar)
        self.q = hyper_config['q']


    def forward(self, k, x, logposterior):
        '''
        k: number of samples
        x: [B,X]
        logposterior(z) -> [P,B]
        '''

        self.B = x.size()[0]

        #Encode
        out = x
        for i in range(len(self.encoder_weights)-1):
            out = self.act_func(self.encoder_weights[i](out))
            # out = self.act_func(self.layer_norms[i].forward(self.encoder_weights[i](out)))

        out = self.encoder_weights[-1](out)
        mean = out[:,:self.z_size]  #[B,Z]
        logvar = out[:,self.z_size:]

        # #Sample
        # eps = Variable(torch.FloatTensor(k, self.B, self.z_size).normal_().type(self.dtype)) #[P,B,Z]
        # z = eps.mul(torch.exp(.5*logvar)) + mean  #[P,B,Z]
        # logqz = lognormal(z, mean, logvar) #[P,B]


        if self.hyper_config['hnf']:
            z, logqz = self.q.sample(mean, logvar, k, logposterior)
        else:
            z, logqz = self.q.sample(mean, logvar, k)

        return z, logqz

















