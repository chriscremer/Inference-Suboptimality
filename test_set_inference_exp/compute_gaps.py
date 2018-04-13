




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


from optimize_local_q import optimize_local_q_dist




from distributions import Gaussian
from distributions import Flow






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





###########################
# Load data

# print ('Loading data')
# with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
#     mnist_data = pickle.load(f, encoding='latin1')
# train_x = mnist_data[0][0]
# valid_x = mnist_data[1][0]
# test_x = mnist_data[2][0]
# train_x = np.concatenate([train_x, valid_x], axis=0)
# print (train_x.shape)

#Load data
print ('Loading data' )
data_location = home + '/Documents/MNIST_data/'
# with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
#     train_x, valid_x, test_x = pickle.load(f)
with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
    train_x, valid_x, test_x = pickle.load(f, encoding='latin1')
print ('Train', train_x.shape)
print ('Valid', valid_x.shape)
print ('Test', test_x.shape)











###########################
# Load model



# this_ckt_file = path_to_save_variables + str(ckt) + '.pt'
# model.load_params(path_to_load_variables=this_ckt_file)
# print ('Init model')
# model = VAE(hyper_config)
# if torch.cuda.is_available():
#     model.cuda()

# print('\nModel:', hyper_config,'\n')


x_size = 784
z_size = 50
# batch_size = 20
# k = 1
#save params 
# start_at = 100
# save_freq = 300
# display_epoch = 3



#small encoder
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }

#no hidden decoder
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                 'decoder_arch': [[z_size,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }

# 2 hidden decoder
hyper_config = { 
                'x_size': x_size,
                'z_size': z_size,
                'act_func': F.tanh,# F.relu,
                'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
                'cuda': 1
            }


# # 4 hidden decoder
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }



q = Gaussian(hyper_config)
# q = Flow(hyper_config)
hyper_config['q'] = q




# Which gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

print ('Init model')
model = VAE(hyper_config)
if torch.cuda.is_available():
    model.cuda()
print('\nModel:', hyper_config,'\n')

print (model.q_dist)
# print (model.q_dist.q)
print (model.generator)


print ('Load params for decoder')
path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/vae_generator_3280.pt'
# path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/decoder_exps/hidden_layers_4_generator_3280.pt'
# path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/decoder_exps/hidden_layers_2_generator_3280.pt'
# path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/decoder_exps/hidden_layers_0_generator_3280.pt'

model.generator.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
print ('loaded variables ' + path_to_load_variables)
print ()



compute_local_opt = 1
compute_amort = 1

compute_local_opt_test = 1
compute_amort_test = 1







if compute_amort:

    print ('Load params for encoder')
    # path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/vae_encoder_100.pt'
    path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/vae_encoder_3280.pt'
    # path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/vae_smallencoder_encoder_3280.pt'
    # path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/vae_regencoder_encoder_3280.pt'
    # path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/vae_smallencoder_withflow_encoder_3280.pt'
    # path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/decoder_exps/hidden_layers_4_encoder_3280.pt'
    # path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/decoder_exps/hidden_layers_2_encoder_3280.pt'
    # path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/decoder_exps/hidden_layers_0_encoder_3280.pt'




    model.q_dist.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
    print ('loaded variables ' + path_to_load_variables)








###########################
# For each datapoint, compute L[q], L[q*], log p(x)

# # log it
# with open(experiment_log, "a") as myfile:
#     myfile.write('Checkpoint' +str(ckt)+'\n')

# start_time = time.time()

n_data = 100 #1000 #100

vaes = []
iwaes = []
vaes_flex = []
iwaes_flex = []



if compute_local_opt:
    print ('optmizing local')
    for i in range(len(train_x[:n_data])):

        print (i)

        x = train_x[i]
        x = Variable(torch.from_numpy(x)).type(model.dtype)
        x = x.view(1,784)

        logposterior = lambda aa: model.logposterior_func2(x=x,z=aa)


        # # flex_model = aux_nf__(model, hyper_config)
        # # if torch.cuda.is_available():
        # #     flex_model.cuda()
        # # vae, iwae = flex_model.train_and_eval(logposterior=logposterior, model=model, x=x)


        # vae, iwae = optimize_local_expressive(logposterior, model, x)
        # print (vae.data.cpu().numpy(),iwae.data.cpu().numpy(),'flex')
        # vaes_flex.append(vae.data.cpu().numpy())
        # iwaes_flex.append(iwae.data.cpu().numpy())

        q_local = Gaussian(hyper_config) #, mean, logvar)
        # q_local = Flow(hyper_config).cuda()#, mean, logvar)

        # print (q_local)

        # vae, iwae = optimize_local_gaussian(logposterior, model, x)
        vae, iwae = optimize_local_q_dist(logposterior, hyper_config, x, q_local)
        print (vae.data.cpu().numpy(),iwae.data.cpu().numpy(),'reg')
        vaes.append(vae.data.cpu().numpy())
        iwaes.append(iwae.data.cpu().numpy())

    print()
    print ('opt vae',np.mean(vaes))
    print ('opt iwae',np.mean(iwaes))
    print()

# print ('opt vae flex',np.mean(vaes_flex))
# print ('opt iwae flex',np.mean(iwaes_flex))
# print()

if compute_amort:
    VAE_train = test_vae(model=model, data_x=train_x[:n_data], batch_size=np.minimum(n_data, 50), display=10, k=5000)
    IW_train = test(model=model, data_x=train_x[:n_data], batch_size=np.minimum(n_data, 50), display=10, k=5000)
    print ('amortized VAE',VAE_train)
    print ('amortized IW',IW_train)


# print()
# AIS_train = test_ais(model=model, data_x=train_x[:n_data], batch_size=n_data, display=2, k=50, n_intermediate_dists=500)
# print ('AIS_train',AIS_train)



# print()
# print()
# print ('AIS_train',AIS_train)
# print()
# print ('opt vae flex',np.mean(vaes_flex))
# # print()
# print ('opt vae',np.mean(vaes))
# # print()
# print ('amortized VAE',VAE_train)
# print()







# TEST SET
print ('TEST SET')

vaes_test = []
iwaes_test = []
# vaes_flex = []
# iwaes_flex = []



if compute_local_opt_test:
    print ('optmizing local')
    for i in range(len(test_x[:n_data])):

        print (i)

        x = test_x[i]
        x = Variable(torch.from_numpy(x)).type(model.dtype)
        x = x.view(1,784)

        logposterior = lambda aa: model.logposterior_func2(x=x,z=aa)


        # # flex_model = aux_nf__(model, hyper_config)
        # # if torch.cuda.is_available():
        # #     flex_model.cuda()
        # # vae, iwae = flex_model.train_and_eval(logposterior=logposterior, model=model, x=x)


        # vae, iwae = optimize_local_expressive(logposterior, model, x)
        # print (vae.data.cpu().numpy(),iwae.data.cpu().numpy(),'flex')
        # vaes_flex.append(vae.data.cpu().numpy())
        # iwaes_flex.append(iwae.data.cpu().numpy())

        q_local = Gaussian(hyper_config) #, mean, logvar)
        # q_local = Flow(hyper_config).cuda()#, mean, logvar)

        # print (q_local)

        # vae, iwae = optimize_local_gaussian(logposterior, model, x)
        vae, iwae = optimize_local_q_dist(logposterior, hyper_config, x, q_local)
        print (vae.data.cpu().numpy(),iwae.data.cpu().numpy(),'reg')
        vaes_test.append(vae.data.cpu().numpy())
        iwaes_test.append(iwae.data.cpu().numpy())

    print()
    print ('opt vae',np.mean(vaes_test))
    print ('opt iwae',np.mean(iwaes_test))
    print()

# print ('opt vae flex',np.mean(vaes_flex))
# print ('opt iwae flex',np.mean(iwaes_flex))
# print()

if compute_amort_test:
    VAE_test = test_vae(model=model, data_x=test_x[:n_data], batch_size=np.minimum(n_data, 50), display=10, k=5000)
    IW_test = test(model=model, data_x=test_x[:n_data], batch_size=np.minimum(n_data, 50), display=10, k=5000)
    print ('amortized VAE',VAE_test)
    print ('amortized IW',IW_test)




print('TRAIN')
print ('opt vae',np.mean(vaes))
print ('opt iwae',np.mean(iwaes))
print ('amortized VAE',VAE_train)
print ('amortized IW',IW_train)


print('TEST')
print ('opt vae',np.mean(vaes_test))
print ('opt iwae',np.mean(iwaes_test))
print ('amortized VAE',VAE_test)
print ('amortized IW',IW_test)






