import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import gc
import bin_utils
from apex.parallel import DistributedDataParallel as DDP
from apex.fp16_utils import *
from apex import amp, optimizers
from apex.multi_tensor_apply import multi_tensor_applier
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='../data/imagenet/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=12, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--arch', type=str, default='latest_cell', help='which architecture to use')
parser.add_argument('--opt_level', type=str, default='O0')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--path_to_weights', type=str, default=None, help='path to the pretrained weights to perform the inference')
parser.add_argument('--num_skip', type=int, default = 3, help='number of conv to not binarize in the first parts of the network')
parser.add_argument('--model_config', type=str, default='bnas-d', help='model config to run inference')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

CLASSES = 1000

configs = {'bnas-d': [12,64, 'O0'], 'bnas-e':[12,68, 'O0'],'bnas-f':[15,68, 'O0'],'bnas-g':[11,74, 'O0'], 'bnas-h':[16,128, 'O1']}
args.layers = configs[args.model_config][0]
args.init_channels=configs[args.model_config][1]
args.opt_level = configs[args.model_config][2]
def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  np.random.seed(args.seed)
  args.distributed = False
  if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

  if args.distributed:
    # FOR DISTRIBUTED:  Set the device according to local_rank.
    args.gpu = args.local_rank

    torch.cuda.set_device(args.gpu)

    # FOR DISTRIBUTED:  Initialize the backend.  torch.distributed.launch will provide
    # environment variables, and requires that you use init_method=`env://`.
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
    args.world_size = torch.distributed.get_world_size()


  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)

  from model_imagenet import NetworkImageNet as Network
  checkpoint = torch.load(args.path_to_weights, map_location = lambda storage, loc: storage.cuda(args.gpu))
  model = Network(args.init_channels, CLASSES, args.layers, genotype).cuda()
  model = amp.initialize(model, opt_level=args.opt_level, keep_batchnorm_fp32=args.keep_batchnorm_fp32,loss_scale=args.loss_scale)
  if args.distributed:
    model = DDP(model,delay_allreduce=True)
  model.load_state_dict(checkpoint['state_dict'])
  amp.load_state_dict(checkpoint['amp'])

  bin_op = bin_utils.BinOp(model, args)
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  val_sampler = None
  if args.distributed:
      val_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)
    
  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, sampler=val_sampler)

  if args.distributed:
    model.module.drop_path_prob = args.drop_path_prob * (args.epochs-1) / args.epochs
  else:
    model.drop_path_prob = args.drop_path_prob * (args.epochs-1) / args.epochs
        
  valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, bin_op)
  logging.info('valid_acc_top1 %f', valid_acc_top1)
  logging.info('valid_acc_top5 %f', valid_acc_top5)
  
def infer(valid_queue, model, criterion, bin_op):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  bin_op.binarization()
  for step, (input, target) in enumerate(valid_queue):
    with torch.no_grad():
      input = input.cuda()
      target = target.cuda(async=True)

      logits = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      if args.distributed:
          reduced_loss = reduce_tensor(loss.data)
          prec1 = reduce_tensor(prec1)
          prec5 = reduce_tensor(prec5)
      else:
          reduced_loss = loss.data
      objs.update(to_python_float(reduced_loss), n)
      top1.update(to_python_float(prec1), n)
      top5.update(to_python_float(prec5), n)

  bin_op.restore()

  return top1.avg, top5.avg, objs.avg

if __name__ == '__main__':
  main() 
