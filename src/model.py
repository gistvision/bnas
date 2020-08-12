import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path


class Cell(nn.Module):

  def __init__(self, genotype,layer_no ,C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    print(C_prev_prev, C_prev, C)
    self.reduction = reduction
    self.layer = layer_no
    self.reduction = reduction
    self.reduction_prev = reduction_prev

    if reduction_prev:
      self.preprocess0 = BinReLUConvBN(C_prev_prev, C,3,2,1, affine=False)
    else:
      self.preprocess0 = BinReLUConvBN(C_prev_prev, C, 1, 1, 0)
    self.preprocess1 = BinReLUConvBN(C_prev, C, 1, 1, 0)
    if self.reduction:
      self.preprocess_res = nn.Sequential(nn.BatchNorm2d(C_prev,affine=True),nn.Conv2d(C_prev,4*C,kernel_size = 2,stride=2))
    if reduction:
      op_names, indices = zip(*genotype.reduce)
      concat = genotype.reduce_concat
    else:
      op_names, indices = zip(*genotype.normal)
      concat = genotype.normal_concat
    self._compile(C, op_names, indices, concat, reduction)

  def _compile(self, C, op_names, indices, concat, reduction):
    assert len(op_names) == len(indices)
    self._steps = len(op_names) // 2
    self._concat = concat
    self.multiplier = len(concat)

    self._ops = nn.ModuleList()
    for name, index in zip(op_names, indices):
      stride = 2 if reduction and index < 2 else 1
      op = OPS[name](C, stride, True)
      self._ops += [op]
    self._indices = indices

  def forward(self, s0, s1, drop_prob):
    res0 = s0
    res1 = s1
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    for i in range(self._steps):
      h1 = states[self._indices[2*i]]
      h2 = states[self._indices[2*i+1]]
      op1 = self._ops[2*i]
      op2 = self._ops[2*i+1]
      h1 = op1(h1)
      h2 = op2(h2)
      if self.training and drop_prob > 0.:
        if not isinstance(op1, Identity):
          h1 = drop_path(h1, drop_prob)
        if not isinstance(op2, Identity):
          h2 = drop_path(h2, drop_prob)
      s = h1 + h2
      states += [s]

    if self.layer == 0:
      return torch.cat([states[i] for i in self._concat], dim=1)
    else: 
      if self.reduction:
        states_out = torch.cat([states[i] for i in self._concat], dim=1)  
        states_out += self.preprocess_res(res1) 
        return states_out
      else:
        states_out = torch.cat([states[i] for i in self._concat], dim=1)
        states_out += res1
        return states_out

class NetworkCIFAR(nn.Module):

  def __init__(self, C, num_classes, layers, genotype):
    super(NetworkCIFAR, self).__init__()
    self._layers = layers

    stem_multiplier = 3
    C_curr = stem_multiplier*C
    self.stem = ReLUConvBN(3, C_curr, 3, stride = 1, padding = 1)
    
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
    self.cells = nn.ModuleList()
    reduction_prev = False
    for i in range(layers):
      if i in [layers//3, 2*layers//3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype,i ,C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier*C_curr

    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def forward(self, input):
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0),-1))
    return logits
