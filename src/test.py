import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from CustomDataParallel import MyDataParallel
from model import NetworkCIFAR as Network
import bin_utils


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--arch', type=str, default='latest_cell_zeroise', help='which architecture to use')
parser.add_argument('--num_skip', type=int, default = 1, help='number of conv to not binarize in the first parts of the network')
parser.add_argument('--parallel', action='store_true', default=False, help='use parallel gpus')
parser.add_argument('--path_to_weights', type=str, default=None, help='path to the pretrained weights to perform the inference')
parser.add_argument('--epochs', type=int, default = 600, help='number of epochs the model was trained for')
args = parser.parse_args()


log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  cudnn.benchmark = True
  cudnn.enabled=True
  logging.info("args = %s", args)
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, genotype)
  if args.parallel:
    model = MyDataParallel(model).cuda() 
  else:
    model = model.cuda()
  bin_op = bin_utils.BinOp(model, args)

  _, valid_transform = utils._data_transforms_cifar10(args)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

  valid_queue = torch.utils.data.DataLoader(
      valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  utils.load(model, args.path_to_weights)
  if args.parallel:
    model.module.drop_path_prob = args.drop_path_prob * (args.epochs-1) / args.epochs
  else:
    model.drop_path_prob = args.drop_path_prob * (args.epochs-1) / args.epochs
  valid_acc, valid_obj = infer(valid_queue, model, criterion, bin_op)
  logging.info('valid_acc %f', valid_acc)

def infer(valid_queue, model, criterion, bin_op):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  bin_op.binarization()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.cuda()
      target = target.cuda()

      logits= model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  bin_op.restore()
  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

