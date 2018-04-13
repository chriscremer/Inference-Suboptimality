


import math
import torch
from torch.autograd import Variable
import numpy as np


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F



# def lognormal(x, mean, logvar):
#     '''
#     x: [B,Z]
#     mean,logvar: [B,Z]
#     output: [B]
#     '''

#     # # D = x.size()[1]
#     # # term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]
#     # term2 = logvar.sum(1) #sum over D, [B]
#     # dif_cov = (x - mean).pow(2)
#     # # dif_cov.div(torch.exp(logvar)) #exp_()) #[P,B,D]
#     # term3 = (dif_cov/torch.exp(logvar)).sum(1) #sum over D, [P,B]
#     # # all_ = Variable(term1) + term2 + term3  #[P,B]
#     # all_ = term2 + term3  #[P,B]
#     # log_N = -.5 * all_
#     # return log_N

#     # term2 = logvar.sum(1) #sum over D, [B]
#     # dif_cov = (x - mean).pow(2)
#     # term3 = (dif_cov/torch.exp(logvar)).sum(1) #sum over D, [P,B]
#     # all_ = term2 + term3  #[P,B]
#     # log_N = -.5 * all_
#     # return log_N

#     # one line 
#     return -.5 * (logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(1))


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    



def lognormal2(x, mean, logvar):
    '''
    x: [P,B,Z]
    mean,logvar: [B,Z]
    output: [P,B]
    '''

    assert len(x.size()) == 3
    assert len(mean.size()) == 2
    assert len(logvar.size()) == 2
    assert x.size()[1] == mean.size()[0]

    D = x.size()[2]

    if torch.cuda.is_available():
        term1 = D * torch.log(torch.cuda.FloatTensor([2.*math.pi])) #[1]
    else:
        term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]


    return -.5 * (Variable(term1) + logvar.sum(1) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))


def lognormal333(x, mean, logvar):
    '''
    x: [P,B,Z]
    mean,logvar: [P,B,Z]
    output: [P,B]
    '''

    assert len(x.size()) == 3
    assert len(mean.size()) == 3
    assert len(logvar.size()) == 3
    assert x.size()[0] == mean.size()[0]
    assert x.size()[1] == mean.size()[1]

    D = x.size()[2]

    if torch.cuda.is_available():
        term1 = D * torch.log(torch.cuda.FloatTensor([2.*math.pi])) #[1]
    else:
        term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]


    return -.5 * (Variable(term1) + logvar.sum(2) + ((x - mean).pow(2)/torch.exp(logvar)).sum(2))



    
def log_bernoulli(pred_no_sig, target):
    '''
    pred_no_sig is [P, B, X] 
    t is [B, X]
    output is [P, B]
    '''

    assert len(pred_no_sig.size()) == 3
    assert len(target.size()) == 2
    assert pred_no_sig.size()[1] == target.size()[0]

    return -(torch.clamp(pred_no_sig, min=0)
                        - pred_no_sig * target
                        + torch.log(1. + torch.exp(-torch.abs(pred_no_sig)))).sum(2) #sum over dimensions








def lognormal3(x, mean, logvar):
    '''
    x: [P]
    mean,logvar: [P]
    output: [1]
    '''

    return -.5 * (logvar.sum(0) + ((x - mean).pow(2)/torch.exp(logvar)).sum(0))




def lognormal4(x, mean, logvar):
    '''
    x: [B,X]
    mean,logvar: [X]
    output: [B]
    '''
    # print x.size()
    # print mean.size()
    # print logvar.size()
    # print mean
    # print logvar
    D = x.size()[1]
    # print D
    term1 = D * torch.log(torch.FloatTensor([2.*math.pi])) #[1]
    # print term1
    # print logvar.sum(0)

    aaa = -.5 * (term1 + logvar.sum(0) + ((x - mean).pow(2)/torch.exp(logvar)).sum(1))
    # print aaa.size()

    return aaa





















































