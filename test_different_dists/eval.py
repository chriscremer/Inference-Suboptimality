







import numpy as np
import gzip
import time
import pickle

from os.path import expanduser
home = expanduser("~")

import sys, os
sys.path.insert(0, '../models')
sys.path.insert(0, '../models/utils')


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


import torch
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from vae_2 import VAE


# from approx_posteriors_v6 import standard
from inference_net import standard

# from ais3 import test_ais


# from optimize_local import optimize_local_gaussian


import csv

from optimize_local_q import optimize_local_q_dist




from distributions import Gaussian
from distributions import Flow
from distributions import HNF





gpu_to_use = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_to_use  #'1'

q_name = sys.argv[2]

hnf = 0
if q_name == 'Gaus':
    q = Gaussian
elif q_name == 'Flow':
    q = Flow
elif q_name == 'HNF':
    q = HNF
    hnf = 1
else:
    dfadfas


# path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/fashion_params/100warm_10k_binarized_fashion_'+q_name #.pt'
path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/fashion_params/10k_binarized_fashion2_SSE_'+q_name #.pt'


# epochs = [100,1000,2200,3280]
# epochs = [400,700]
# epochs = [20,100,200,300,360]
# epochs = [100,300,500,700,1000]
epochs = [100,300,600,700]
# epochs = [360]



n_data =1006


# write to
file_ = home+'/Documents/tmp/inference_suboptimality/over_training_exps/results_'+str(n_data)+'_10k_fashion_binarized2_SSE.txt'








def test_vae(model, data_x, batch_size, display, k):
    
    time_ = time.time()
    elbos = []
    data_index= 0
    for i in range(int(len(data_x)/ batch_size)):

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size

        batch = Variable(torch.from_numpy(batch)).type(model.dtype)

        elbo, logpxz, logqz = model.forward2(batch, k=k)

        elbos.append(elbo.data[0])

        # if i%display==0:
        #     print (i,len(data_x)/ batch_size, np.mean(elbos))

    mean_ = np.mean(elbos)
    # print(mean_, 'T:', time.time()-time_)

    return mean_#, time.time()-time_





def test(model, data_x, batch_size, display, k):
    
    time_ = time.time()
    elbos = []
    data_index= 0
    for i in range(int(len(data_x)/ batch_size)):

        batch = data_x[data_index:data_index+batch_size]
        data_index += batch_size

        batch = Variable(torch.from_numpy(batch)).type(model.dtype)

        elbo, logpxz, logqz = model(batch, k=k)

        elbos.append(elbo.data[0])

        # if i%display==0:
        #     print (i,len(data_x)/ batch_size, np.mean(elbos))

    mean_ = np.mean(elbos)
    # print(mean_, 'T:', time.time()-time_)

    return mean_#, time.time()-time_











#FASHION
def load_mnist(path, kind='train'):

    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(-1, 784)

    return images#, labels


path = home+'/Documents/fashion_MNIST'

train_x = load_mnist(path=path)
test_x = load_mnist(path=path, kind='t10k')

train_x = train_x / 255.
test_x = test_x / 255.

#binarize
train_x = (train_x > .5).astype(float)
test_x = (test_x > .5).astype(float)


print (train_x.shape)
print (test_x.shape)
print ()

valid_x = train_x[50000:]
# train_x = train_x[:50000]
train_x = train_x[:10000]  #small dataset 


print (train_x.shape)
print (valid_x.shape)
print (test_x.shape)
print ()





x_size = 784
z_size = 20
batch_size = 50
k = 1
#save params 
start_at = 100
save_freq = 300
display_epoch = 3

# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.elu, #F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1,
#                 'hnf': hnf
#             }



# #SE
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.elu, #F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,100],[100,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1,
#                 'hnf': hnf
#             }


#SSE
hyper_config = { 
                'x_size': x_size,
                'z_size': z_size,
                'act_func': F.elu, #F.tanh,# F.relu,
                'encoder_arch': [[x_size,z_size*2]],
                'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
                'cuda': 1,
                'hnf': hnf
            }



hyper_config['q'] = q(hyper_config)


print ('Init model')
model = VAE(hyper_config)
if torch.cuda.is_available():
    model.cuda()

print('\nModel:', hyper_config,'\n')









batch_size = 4

k = 5000

for epoch in epochs:

    print (epoch, epochs)



    print ('Load params for decoder')
    path_to_load_variables=path_to_save_variables+'_generator_'+str(epoch)+'.pt'
    model.generator.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
    print ('loaded variables ' + path_to_load_variables)
    # print ()


    print ('Load params for encoder')
    path_to_load_variables=path_to_save_variables+'_encoder_'+str(epoch)+'.pt'
    model.q_dist.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
    print ('loaded variables ' + path_to_load_variables)




    #TRAINING SET AMORT

    VAE_train = test_vae(model=model, data_x=train_x[:n_data], batch_size=np.minimum(n_data, batch_size), display=10, k=k)
    IW_train = test(model=model, data_x=train_x[:n_data], batch_size=np.minimum(n_data, batch_size), display=10, k=k)
    print ('train amortized VAE',VAE_train)
    print ('train amortized IW',IW_train)


    with open(file_, 'a') as f:
        writer = csv.writer(f, delimiter=' ')

        writer.writerow([q_name,'training', epoch, 'L_q', VAE_train])
        writer.writerow([q_name,'training', epoch, 'L_q_IWAE', IW_train])



    #TEST SET AMORT

    VAE_test = test_vae(model=model, data_x=test_x[:n_data], batch_size=np.minimum(n_data, batch_size), display=10, k=k)
    IW_test = test(model=model, data_x=test_x[:n_data], batch_size=np.minimum(n_data, batch_size), display=10, k=k)
    print ('test amortized VAE',VAE_test)
    print ('test amortized IW',IW_test)
    print()


    with open(file_, 'a') as f:
        writer = csv.writer(f, delimiter=' ')

        writer.writerow([q_name,'validation', epoch, 'L_q', VAE_test])
        writer.writerow([q_name,'validation', epoch, 'L_q_IWAE', IW_test])




# values = {}
# values['training'] = {}
# values['validation'] = {}

# max_value = None
# min_value = None


# #Get numbder of distributions and epochs and datasets
# datasets = []
# distributions = []
# epochs = []
# bounds = []

# with open(file_, 'r') as f:
#     reader = csv.reader(f, delimiter=' ')
#     for row in reader:
#         if len(row):

#             distribution = row[0]
#             dataset = row[1]
#             epoch = row[2]
#             bound = row[2]

#             if distribution not in distributions:
#                 distributions.append(distribution)

#          and row[1] in ['training','validation']: 
#             # print (row)

#             value = row[3]

#             if epoch not in values[dataset]:
#                 values[dataset][epoch] = {}
#                 if epoch not in epochs:
#                     epochs.append(epoch)
#                     print (epoch)

#             values[dataset][epoch][bound] = value

#             if max_value == None or float(value) > max_value:
#                 max_value = float(value)
#             if min_value == None or float(value) < min_value:
#                 min_value = float(value)
















