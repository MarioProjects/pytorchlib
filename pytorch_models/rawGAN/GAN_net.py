import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import Linear_act,return_activation,apply_conv,apply_linear,apply_pool, MamasitaNetwork,add_gaussian,apply_DePool,apply_DeConv		
import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import Linear_act,return_activation,apply_conv,apply_linear,apply_pool, MamasitaNetwork,add_gaussian,apply_DePool,apply_DeConv		


class GAN(MamasitaNetwork):
    def __init__(self):
        super(GAN,self).__init__()
        self.__init_generator__()
        self.__init_discriminator__()
        self.CE=nn.BCELoss()
        #self.CE=nn.CrossEntropyLoss()

    def __init_generator__(self):
        fc_l1=apply_linear(100,128,'relu',bn=False)#,std=0.3,shape=(100,1024))
        fc_l2=apply_linear(128,784,'sigmoid',bn=False)#,std=0.3,shape=(100,128))

        self.generator_forward=nn.Sequential(fc_l1,fc_l2)
        self.generator_sampler=torch.zeros(50,100).cuda()

    def __init_discriminator__(self):
        fc_l1=apply_linear(784,128,'relu',bn=False)#,std=0.3,shape=(100,1024))
        fc_l2=apply_linear(128,1,'sigmoid',bn=False)#,std=0.3,shape=(100,128))


        self.discriminator_forward=nn.Sequential(fc_l1,fc_l2)

    def sample_from_z(self):
        #return torch.randn((50,100)).cuda()
        return self.generator_sampler.normal_(0,1)


    def forward_generator(self,z):	
        self.train()
        return self.generator_forward(z)
    
    def forward_discriminator(self,x):
        self.train()
        return self.discriminator_forward(z)

    def discriminator_cost(self,x,t):
        return self.CE(x,t)