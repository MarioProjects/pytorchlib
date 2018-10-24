import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torchtensorvision import transforms as tensor_transforms
import torchvision.datasets  as dset

#python
import os
import sys
sys.path.extend(['../../../'])
import logging
import numpy

#utils
from utils import select_optimizer

#data
import load_data
from load_data import create_dataset,return_samples,load_mnist

#neural networks
from GAN_net import GAN

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Baseline models for mnist in Pytorch by Juan Maron~as Molano. Resarcher at PRHLT Universidad Politecnica de Valencia jmaronasm@prhlt.upv.es . Enjoy!')
    '''String Variables'''
    parser.add_argument('--epochs', type=int,nargs='+',required=True,help='number of epochs')
    parser.add_argument('--lr', type=float,nargs='+',required=True,help='learning rate')
    parser.add_argument('--mmu', type=float,nargs='+',required=True,help='momentum')
    parser.add_argument('--anneal', type=str,required=False,help='linear anneal')
    parser.add_argument('--optim', type=str,required=True,choices=['ADAM','SGD'],help='optimizer')
    parser.add_argument('--n_gpu', type=int,required=True,help='which gpu to use')
    parser.add_argument('--seed', type=int,default=1,required=False,help='random_seed')

    aux=parser.parse_args()
    torch.manual_seed(seed=aux.seed)
    torch.cuda.manual_seed(seed=aux.seed)	
    numpy.random.seed(seed=aux.seed)

    torch.cuda.set_device(aux.n_gpu)

    arguments=[aux.epochs,aux.lr,aux.mmu,aux.optim,aux.anneal,aux.seed]
    return arguments

def anneal_lr(lr_init,epochs_N,e):
        lr_new=-(lr_init/epochs_N)*e+lr_init
        return lr_new


if __name__=='__main__':
    '''
    Script for training a variatonal autoencoder on MNIST on Fully Connected to a z space of 784 dimensions.
    '''
    #training
    epochs_t,lr_t,mmu_t,optim,anneal,seed=parse_args()
    activate_anneal=False
    linear_anneal=False
    #training
    assert len(mmu_t)==len(epochs_t) and len(mmu_t)==len(lr_t)
    torch.backends.cudnn.enabled=True
    
    #for saving the model and logging
    counter=0

    '''
    Dataset
    '''
    trans = transforms.Compose([transforms.ToTensor()])
    trans_test = transforms.Compose([transforms.ToTensor()])

    train_set = dset.MNIST(root='./',transform=trans, train=True, download=True)
    test_set = dset.MNIST(root='./',transform=trans_test, train=False, download=True)

    batch_size = 50

    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)

    counter=0
    directory="./model/"+"/seed_"+str(seed)+"/"+str(counter)+"/"
    while True:
        if os.path.isdir(directory):
            counter+=1
            directory="./model/"+"/seed_"+str(seed)+"/"+str(counter)+"/"
        else:
            break

    model_dir = directory+'models/'
    log_dir =directory+'logs/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(filename=log_dir+'train.log',level=logging.INFO)
    logging.info("Logger for model: {}".format('GANs:'))
    logging.info("Training specificacitions: epochs {} lr {} mmu {} ".format(epochs_t,lr_t,mmu_t))

    Net=GAN().cuda()
    D_parameters=[]
    G_parameters=[]

    for name,param in Net.named_parameters():
        if 'discriminator' in name:
            D_parameters+=[param]
        else:
            G_parameters+=[param]

    G_t=torch.zeros(50,1).cuda()
    D_t=torch.ones(50,1).cuda()
    total_ep=0
    for ind,(mmu,l,ep) in enumerate(zip(mmu_t,lr_t,epochs_t)):
        lr_new=l
        D_optim=select_optimizer(D_parameters,lr=1e-3,mmu=mmu,optim='ADAM')
        G_optim=select_optimizer(G_parameters,lr=1e-3,mmu=mmu,optim='ADAM')
        '''
        Activate annealing
        '''
        if ind == len(epochs_t)-1 and linear_anneal:
                        activate_anneal=True
                        lr_init=l
                        epochs_N=ep
                        mmu_init=mmu
        for e in range(ep):

            '''lr anneal'''
            if activate_anneal:
                l_new=anneal_lr(lr_init,epochs_N,e)
                D_optim=select_optimizer(D_parameters,lr=1e-3,mmu=mmu,optim='ADAM')
                G_optim=select_optimizer(G_parameters,lr=1e-3,mmu=mmu,optim='ADAM')
            
            Dcost_acc=torch.tensor(0.0).cuda()
            Gcost_acc=torch.tensor(0.0).cuda()
            for x,t in train_loader:
                #use 50 true label and 50 generated
                x=x.cuda().view(-1,28*28)

                #discriminator updates
                z=Net.sample_from_z()#sample from latent
                x_fake = Net.generator_forward(z)# xfake=G(z)				
                D_real_fake=Net.discriminator_forward(torch.cat((x,x_fake),0))# t=D([xfake,x])
                
                D_cost=Net.discriminator_cost(D_real_fake,torch.cat((D_t,G_t),0))#cross entropy

                D_optim.zero_grad()#clean grads
                G_optim.zero_grad()

                D_cost.backward()#backward 
                D_optim.step()#update discriminator. MINIMIZE CROSS ENTROPY (maximize likelihood)

                D_optim.zero_grad()#reset grads again for non overlap
                G_optim.zero_grad()

                #generator updates
                z=Net.sample_from_z()#sample from latent
                x_fake = Net.generator_forward(z)# xfake=G(z)
                D_fake=Net.discriminator_forward(x_fake)# t=D(G(z))
                G_cost=Net.discriminator_cost(D_fake,D_t)#cross entropy. CHANGE SIGN. we minimize likelihood which is maximize cross entropy
                G_cost.backward()#backward 
                G_optim.step()#update discriminator. MINIMIZE CROSS ENTROPY (maximize likelihood)
            

                Dcost_acc+=D_cost.data
                Gcost_acc+=G_cost.data

            #sample some images from the generator			
            z=Net.sample_from_z()#sample from latent
            x_fake = Net.generator_forward(z)# xfake=G(z)	
            save_image(x_fake.view(50,1,28,28).cpu(),'./images/itet_{}.png'.format(total_ep))

            print ("On epoch {} Discriminator cost {}  Generator cost {}".format(total_ep,Dcost_acc,Gcost_acc))
            total_ep+=1

    

            #print("On epoch {} with lr {} LLH train conv {:.3f} LLH train fc {:.3f} SSE train fc {:.3f} train error conv {:.3f} train error fc {:.3f} test error conv {:.3f} test error fc {:.3f}  of total train samples {} total test samples {}".format(e,lr_new,ceerr_train_conv,ceerr_train_fc,sse_train,mcerr_train_conv/tot_train*100,mcerr_train_fc/tot_train*100,mcerr_test_conv/tot_test*100,mcerr_test_fc/tot_test*100,tot_train,tot_test))
            

            #torch.save(baseline_net,model_dir+'baseline_model')