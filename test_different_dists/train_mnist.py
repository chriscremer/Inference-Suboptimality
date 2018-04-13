


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

# from ais2 import test_ais

# from pytorch_vae_v6 import VAE

# from vae_1 import VAE
from vae_2 import VAE


# from approx_posteriors_v6 import FFG_LN
# from approx_posteriors_v6 import ANF_LN
# import argparse
# from approx_posteriors_v6 import standard
from inference_net import standard

from distributions import Gaussian
from distributions import Flow
from distributions import HNF
from distributions import Flow1





gpu_to_use = sys.argv[1]
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_to_use  #'1'

q_name = sys.argv[2]

hnf = 0
if q_name == 'Gaus':
    q = Gaussian
elif q_name == 'Flow':
    q = Flow
elif q_name == 'Flow1':
    q = Flow1
elif q_name == 'HNF':
    q = HNF
    hnf = 1
else:
    dfadfas




# path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/fashion_params/10k_binarized_fashion2_SSE_'+q_name #.pt'



# path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/fashion_params/binarized_fashion3_LE_'+q_name #.pt'

path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/fashion_params/binarized_fashion3_'+q_name #.pt'








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
train_x = train_x[:50000]
# train_x = train_x[:10000]  #small dataset 


print (train_x.shape)
print (valid_x.shape)
print (test_x.shape)
print ()


# fdsa




# print (train_x)
# fads

# print (np.max(train_x))
# print (test_x[3])
# fsda


# print ('Loading data')
# with open(home+'/Documents/MNIST_data/mnist.pkl','rb') as f:
#     mnist_data = pickle.load(f, encoding='latin1')
# train_x = mnist_data[0][0]
# valid_x = mnist_data[1][0]
# test_x = mnist_data[2][0]
# train_x = np.concatenate([train_x, valid_x], axis=0)
# print (train_x.shape)



# #Load data  mnist
# print ('Loading data' )
# data_location = home + '/Documents/MNIST_data/'
# # with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
# #     train_x, valid_x, test_x = pickle.load(f)
# with open(data_location + 'binarized_mnist.pkl', 'rb') as f:
#     train_x, valid_x, test_x = pickle.load(f, encoding='latin1')
# print ('Train', train_x.shape)
# print ('Valid', valid_x.shape)
# print ('Test', test_x.shape)


# print (np.max(train_x))

# fadad




def train_encoder_and_decoder(model, train_x, test_x, k, batch_size,
                    start_at, save_freq, display_epoch, 
                    path_to_save_variables):

    train_y = torch.from_numpy(np.zeros(len(train_x)))
    train_x = torch.from_numpy(train_x).float().type(model.dtype)

    train_ = torch.utils.data.TensorDataset(train_x, train_y)
    train_loader = torch.utils.data.DataLoader(train_, batch_size=batch_size, shuffle=True)

    #IWAE paper training strategy
    time_ = time.time()
    total_epochs = 0

    i_max = 7
    # i_max = 6

    warmup_over_epochs = 100.
    # warmup_over_epochs = 20.


    all_params = []
    for aaa in model.q_dist.parameters():
        all_params.append(aaa)
    for aaa in model.generator.parameters():
        all_params.append(aaa)
    # print (len(all_params), 'number of params')

    print (model.q_dist)
    # print (model.q_dist.q)
    print (model.generator)

    # fads


    for i in range(0,i_max+1):

        lr = .001 * 10**(-i/float(i_max))
        print (i, 'LR:', lr)

        # # optimizer = optim.Adam(model.parameters(), lr=lr)
        # print (model.q_dist)
        # print (model.generator)
        # print (model.q_dist.parameters())
        # print (model.generator.parameters())

        # print ('Encoder')
        # for aaa in model.q_dist.parameters():
        #     # print (aaa)
        #     print (aaa.size())
        # print ('Decoder')
        # for aaa in model.generator.parameters():
        #     # print (aaa)
        #     print (aaa.size())
        # # fasdfs
        # fads


        optimizer = optim.Adam(all_params, lr=lr)

        epochs = 3**(i)

        for epoch in range(1, epochs + 1):

            for batch_idx, (data, target) in enumerate(train_loader):

                batch = Variable(data)#.type(model.dtype)

                optimizer.zero_grad()

                warmup = total_epochs/warmup_over_epochs
                if warmup > 1.:
                    warmup = 1.

                elbo, logpxz, logqz = model.forward(batch, k=k, warmup=warmup)

                loss = -(elbo)
                loss.backward()
                optimizer.step()

            total_epochs += 1


            if total_epochs%display_epoch==0:
                print ('Train Epoch: {}/{}'.format(epoch, epochs),
                    'total_epochs {}'.format(total_epochs),
                    'LL:{:.3f}'.format(-loss.data[0]),
                    'logpxz:{:.3f}'.format(logpxz.data[0]),
                    'logqz:{:.3f}'.format(logqz.data[0]),
                    'warmup:{:.3f}'.format(warmup),
                    'T:{:.2f}'.format(time.time()-time_),
                    )
                time_ = time.time()


            if total_epochs >= start_at and (total_epochs-start_at)%save_freq==0:

                # save params
                save_file = path_to_save_variables+'_encoder_'+str(total_epochs)+'.pt'
                torch.save(model.q_dist.state_dict(), save_file)
                print ('saved variables ' + save_file)
                save_file = path_to_save_variables+'_generator_'+str(total_epochs)+'.pt'
                torch.save(model.generator.state_dict(), save_file)
                print ('saved variables ' + save_file)



    # save params
    save_file = path_to_save_variables+'_encoder_'+str(total_epochs)+'.pt'
    torch.save(model.q_dist.state_dict(), save_file)
    print ('saved variables ' + save_file)
    save_file = path_to_save_variables+'_generator_'+str(total_epochs)+'.pt'
    torch.save(model.generator.state_dict(), save_file)
    print ('saved variables ' + save_file)


    print ('done training')













# Which gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


x_size = 784
z_size = 20
batch_size = 50
k = 1
#save params 
# start_at = 50
# save_freq = 250
start_at = 100
save_freq = 300

display_epoch = 3

hyper_config = { 
                'x_size': x_size,
                'z_size': z_size,
                'act_func': F.elu, #F.tanh,# F.relu,
                'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
                'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
                'cuda': 1,
                'hnf': hnf
            }


# #LB
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.elu, #F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,500],[500,500],[500,z_size*2]],
#                 'decoder_arch': [[z_size,500],[500,500],[500,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1,
#                 'hnf': hnf
#             }


# #LD
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.elu, #F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                 'decoder_arch': [[z_size,500],[500,500],[500,500],[500,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1,
#                 'hnf': hnf
#             }




# #LE
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.elu, #F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,500],[500,500],[500,500],[500,z_size*2]],
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




# #SSE
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.elu, #F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1,
#                 'hnf': hnf
#             }



# #LE
# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.elu, #F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,500],[500,500],[500,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }

# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                 'decoder_arch': [[z_size,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }


# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }


# q = Gaussian(hyper_config)
# # q = Flow(hyper_config)
hyper_config['q'] = q(hyper_config)


print ('Init model')
model = VAE(hyper_config)
if torch.cuda.is_available():
    model.cuda()

print('\nModel:', hyper_config,'\n')




# path_to_load_variables=''
# path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/fashion_params/LE_binarized_fashion' #.pt'
# path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/fashion_params/binarized_fashion_' #.pt'

# path_to_save_variables=home+'/Documents/tmp/pytorch_vae'+str(epochs)+'.pt'
# path_to_save_variables=this_dir+'/params_'+model_name+'_'
# path_to_save_variables=''



print('\nTraining')
# train_lr_schedule(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size,
#                     start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
#                     path_to_save_variables=path_to_save_variables)


train_encoder_and_decoder(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size,
                    start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
                    path_to_save_variables=path_to_save_variables)

print ('Done.')






































