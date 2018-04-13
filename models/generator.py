



import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F









class Generator(nn.Module):

    def __init__(self, hyper_config):
        super(Generator, self).__init__()

        if hyper_config['cuda']:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.z_size = hyper_config['z_size']
        self.x_size = hyper_config['x_size']
        self.act_func = hyper_config['act_func']

        #Decoder
        self.decoder_weights = []
        self.layer_norms = []
        for i in range(len(hyper_config['decoder_arch'])):
            self.decoder_weights.append(nn.Linear(hyper_config['decoder_arch'][i][0], hyper_config['decoder_arch'][i][1]))

        count =1
        for i in range(len(self.decoder_weights)):
            self.add_module(str(count), self.decoder_weights[i])
            count+=1
   

    def decode(self, z):
        k = z.size()[0]
        B = z.size()[1]
        z = z.view(-1, self.z_size)

        out = z
        for i in range(len(self.decoder_weights)-1):
            out = self.act_func(self.decoder_weights[i](out))
            # out = self.act_func(self.layer_norms[i].forward(self.decoder_weights[i](out)))
        out = self.decoder_weights[-1](out)

        x = out.view(k, B, self.x_size)
        return x








