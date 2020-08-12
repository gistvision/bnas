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
      if self.reduction:
        self.preprocess0_red = BinReLUConvBN(C_prev_prev, C, 3, 2,1)
      else:
        self.preprocess0 = BinReLUConvBN(C_prev_prev, C, 3,2,1)
    else:
      if self.reduction:
        self.preprocess0_red = BinReLUConvBN(C_prev_prev, C ,1,1,0)
      else:
        self.preprocess0 = BinReLUConvBN(C_prev_prev, C, 1, 1, 0)
    if self.reduction:
      self.preprocess1_red = BinReLUConvBN(C_prev, C, 1,1,0)
    else:
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
    self.dropout = nn.Dropout(0.2)

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
    if self.reduction:
      s0 = self.preprocess0_red(res0)
      s1 = self.preprocess1_red(res1)
    else:
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
         #states_out.append(states[0])
         #print (states_out)
         #print (self.res(res1),'res')   
        states_out += self.preprocess_res(res1) 
        return states_out
      else:
          #if self.layer>0:
        states_out = torch.cat([states[i] for i in self._concat], dim=1)
           #states_out.append(states[1])
        states_out += res1
        return states_out

class NetworkImageNet(nn.Module):

  def __init__(self, C, num_classes, layers, genotype):
    super(NetworkImageNet, self).__init__()
    self._layers = layers
    self.stem0 = nn.Sequential(
      nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C // 2),
      nn.ReLU(),
      nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    self.stem1 = nn.Sequential(
      nn.ReLU(),
      nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
      nn.BatchNorm2d(C),
    )

    C_prev_prev, C_prev, C_curr = C, C, C

    self.cells = nn.ModuleList()
    reduction_prev = True
    for i in range(layers):
      if i in [layers // 3, 2 * layers // 3]:
        C_curr *= 2
        reduction = True
      else:
        reduction = False
      cell = Cell(genotype,i ,C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
      reduction_prev = reduction
      self.cells += [cell]
      C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

    self.global_pooling = nn.AvgPool2d(7)
    self.classifier = nn.Linear(C_prev, num_classes)
    self.dropout=nn.Dropout(0.5)
  def forward(self, input):
    logits_aux = None
    s0 = self.stem0(input)
    s1 = self.stem1(s0)
    for i, cell in enumerate(self.cells):
      s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

    out = self.global_pooling(s1)
    logits = self.classifier(out.view(out.size(0), -1))
    return logits

