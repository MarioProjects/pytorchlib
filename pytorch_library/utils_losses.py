
import types
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd.variable import Variable



def _to_one_hot(y, n_dims, dtype=torch.cuda.FloatTensor):
    scatter_dim = len(y.size())
    y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), n_dims).type(dtype)
        
    return zeros.scatter(scatter_dim, y_tensor, 1)


"""
    Improving Pairwise Ranking for Multi-label Image Classification
    https://arxiv.org/pdf/1704.03135.pdf
    from: https://gist.github.com/NegatioN/eb2c23fc98e67a6396f6ea77e384c549
"""


class LSEP2(torch.autograd.Function): 
    """
    Autograd function of LSEP loss. Appropirate for multi-label
    - Reference: Li+2017
      https://arxiv.org/pdf/1704.03135.pdf
      Code-ref: https://github.com/Mipanox/Bird_cocktail/blob/196e9404a4f7022d1e56433112f581b34a334e53/utils.py#L332
    """
    
    @staticmethod
    def forward(ctx, input, target):
        target=_to_one_hot(target,340)
        batch_size = target.size()[0]
        label_size = target.size()[1]

        ##
        positive_indices = target.gt(0).float()
        negative_indices = target.eq(0).float()
        
        ## summing over all negatives and positives
        loss = 0.
        for i in range(input.size()[0]):
            pos = positive_indices[i].nonzero()
            neg = negative_indices[i].nonzero()
            pos_examples = input[i, pos]
            neg_examples = torch.transpose(input[i, neg], 0, 1)
            loss += torch.sum(torch.exp(neg_examples - pos_examples))
        
        loss = torch.log(1 + loss)
        
        ctx.save_for_backward(input, target)
        ctx.loss = loss
        ctx.positive_indices = positive_indices
        ctx.negative_indices = negative_indices
        
        return loss

    # This function has only a single output, so it gets only one gradient 
    @staticmethod
    def backward(ctx, grad_output):
        dtype = torch.cuda.FloatTensor
        input, target = ctx.saved_variables
        N = input.size()[1]
        loss = Variable(ctx.loss, requires_grad = False)
        positive_indices = ctx.positive_indices
        negative_indices = ctx.negative_indices

        fac  = -1 / loss
        grad_input = torch.zeros(input.size()).type(dtype)
        
        scale = grad_input.size(0), -1
        phot = _to_one_hot(positive_indices.nonzero()[:, 1].view(*scale), N)
        nhot = _to_one_hot(negative_indices.nonzero()[:, 1].view(*scale), N)

        scale = (len(phot), *nhot.size())
        diffs = torch.sum(phot - nhot.expand(scale), dim=2)
        grads_input = (Variable(diffs * torch.exp(-input * diffs)) * (grad_output * fac))
        
        return grad_input, None, None
    
#--- main class
class LSEPLoss2(nn.Module): 
    def __init__(self): 
        super(LSEPLoss2, self).__init__()
        
    def forward(self, input, target): 
        return LSEP2.apply(input, target)
    
def loss_lsep2(outputs, labels):
    return LSEPLoss2()(F.sigmoid(outputs), _to_one_hot(labels, len(categories)))