import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.lamda = args.lamda
    self.tau = args.tau


  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer,step):
    self.optimizer.zero_grad()
    self._backward_step(input_valid, target_valid,step)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid,step):
    normal_alphas = F.softmax(self.model.arch_parameters()[0], dim = -1)
    reduce_alphas = F.softmax(self.model.arch_parameters()[1], dim = -1)
    total_entropy = 0.0
    for i in range(normal_alphas.shape[0]):
        temp = -sum(normal_alphas[i,:]*torch.log(normal_alphas[i,:])).cuda()
        total_entropy += temp 
    for j in range(reduce_alphas.shape[0]):
        temp2 = -sum(reduce_alphas[i,:]*torch.log(reduce_alphas[i,:])).cuda()
        total_entropy += temp2            

    total_entropy = total_entropy/ (normal_alphas.shape[0] + reduce_alphas.shape[0])  
    loss = self.model._loss(input_valid, target_valid) - self.lamda*total_entropy* np.exp(-step/self.tau)    
    loss.backward()



