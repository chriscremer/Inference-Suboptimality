


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

from inference_net import standard

from distributions import Gaussian
from distributions import Flow1



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










def train_encdoer_and_decoder(model, train_x, test_x, k, batch_size,
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

    warmup_over_epochs = 100.


    all_params = []
    for aaa in model.q_dist.parameters():
        all_params.append(aaa)
    # for aaa in model.generator.parameters():
    #     all_params.append(aaa)
    # print (len(all_params), 'number of params')

    print (model.q_dist)
    # print (model.q_dist.q)
    print (model.generator)

    for i in range(0,i_max+1):

        lr = .001 * 10**(-i/float(i_max))
        print (i, 'LR:', lr)

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
                # save_file = path_to_save_variables+'_generator_'+str(total_epochs)+'.pt'
                # torch.save(model.generator.state_dict(), save_file)
                # print ('saved variables ' + save_file)



    # save params
    save_file = path_to_save_variables+'_encoder_'+str(total_epochs)+'.pt'
    torch.save(model.q_dist.state_dict(), save_file)
    print ('saved variables ' + save_file)
    # save_file = path_to_save_variables+'_generator_'+str(total_epochs)+'.pt'
    # torch.save(model.generator.state_dict(), save_file)
    # print ('saved variables ' + save_file)


    print ('done training')











x_size = 784
z_size = 50
batch_size = 20
k = 1
#save params 
start_at = 100
save_freq = 300
display_epoch = 3

# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,200],[200,200],[200,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }




# Which gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0'





# hyper_config = { 
#                 'x_size': x_size,
#                 'z_size': z_size,
#                 'act_func': F.tanh,# F.relu,
#                 'encoder_arch': [[x_size,z_size*2]],
#                 'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
#                 'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
#                 'cuda': 1
#             }


#flow1
hyper_config = { 
                'x_size': x_size,
                'z_size': z_size,
                'act_func': F.tanh,  #F.elu, #,# F.relu,
                'encoder_arch': [[x_size,z_size*2]],
                'decoder_arch': [[z_size,200],[200,200],[200,x_size]],
                'q_dist': standard, #FFG_LN#,#hnf,#aux_nf,#flow1,#,
                'cuda': 1,
                'hnf': 0
            }

hyper_config['q'] = Flow1(hyper_config)
# hyper_config['q'] = Gaussian(hyper_config)


print ('Init model')
model = VAE(hyper_config)
if torch.cuda.is_available():
    model.cuda()

print('\nModel:', hyper_config,'\n')





# path_to_load_variables=''
path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/vae_smallencoder_withflow1' #.pt'
# path_to_save_variables=home+'/Documents/tmp/inference_suboptimality/vae_regencoder' #.pt'
# path_to_save_variables=home+'/Documents/tmp/pytorch_vae'+str(epochs)+'.pt'
# path_to_save_variables=this_dir+'/params_'+model_name+'_'
# path_to_save_variables=''


# load generator
print ('Load params for decoder')
path_to_load_variables=home+'/Documents/tmp/inference_suboptimality/vae_generator_3280.pt'
model.generator.load_state_dict(torch.load(path_to_load_variables, map_location=lambda storage, loc: storage))
print ('loaded variables ' + path_to_load_variables)
print ()








print('\nTraining')
# train_lr_schedule(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size,
#                     start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
#                     path_to_save_variables=path_to_save_variables)


train_encdoer_and_decoder(model=model, train_x=train_x, test_x=test_x, k=k, batch_size=batch_size,
                    start_at=start_at, save_freq=save_freq, display_epoch=display_epoch, 
                    path_to_save_variables=path_to_save_variables)

print ('Done.')





































