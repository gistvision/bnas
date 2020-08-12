import torch
import torch.nn as nn

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'bin_dil_conv_3x3' : lambda C, stride, affine: BinDilConv(C, C, 3, stride, 2, 2, affine=affine),
  'bin_dil_conv_5x5' : lambda C, stride, affine: BinDilConv(C, C, 5, stride, 4, 2, affine=affine),
  'bin_conv_3x3' : lambda C, stride, affine: BinReLUConvBN(C, C, 3, stride, 1, affine=affine),
  'bin_conv_5x5' : lambda C, stride, affine: BinReLUConvBN(C, C, 5, stride, 2, affine=affine) 
}

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(BinReLUConvBN, self).__init__()
    self.bn = nn.BatchNorm2d(C_in, affine=affine)
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.ReLU(inplace=False),     
    )

  def forward(self, x):
     
    x = self.bn(x)
    x,_ = BinActive()(x)       
    return self.op(x)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.bn = nn.BatchNorm2d(C_in, affine=affine)
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.ReLU(inplace=False),     
    )

  def forward(self, x):  
    x = self.bn(x)  
    return self.op(x)


class BinDilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(BinDilConv, self).__init__()
    self.bn =  nn.BatchNorm2d(C_in, affine=affine)
    self.op = nn.Sequential(
      nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
     
       nn.ReLU(inplace=False),
      )

  def forward(self, x):
    x = self.bn(x)
    x, _ = BinActive()(x)
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


