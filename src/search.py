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
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import bin_utils_search

from torch.autograd import Variable
from model_search import Network
from architect import Architect


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--lamda', type = float, default = 1.0, help ='multiplicative factor in er')
parser.add_argument('--tau', type =float, default = 7.7, help='decay factor in er')
parser.add_argument('--num_skip', type=int, default = 1, help='number of conv to not binarize in the first parts of the network')
parser.add_argument('--gamma', type=float, default =3, help ='transferability hyper-parameter for Zeroise')
args = parser.parse_args()
args.geno_name = args.save
args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info("args = %s", args)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  train_transform, valid_transform = utils._data_transforms_cifar10(args)

  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
  test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  test_queue = torch.utils.data.DataLoader(
      test_data, batch_size=args.batch_size,
      shuffle=False, pin_memory=True, num_workers=2)


  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)



  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)
  bin_op = bin_utils_search.BinOp(model, args)
  best_acc = 0.
  best_genotypes = []
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype()
    genotype_img = model.genotype(args.gamma)
    logging.info('genotype = %s', genotype)
    logging.info(F.softmax(model.alphas_normal, dim=-1))
    logging.info(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, bin_op, epoch)
    logging.info('train_acc %f', train_acc)

    # validation
    valid_acc, valid_obj = infer(valid_queue, model, criterion, bin_op)
    logging.info('valid_acc %f', valid_acc)
    if best_acc < valid_acc:
      best_acc = valid_acc
      if len(best_genotypes) > 0:
        best_genotypes[0] = genotype
        best_genotypes[1] = genotype_img
      else:
        best_genotypes.append(genotype)
        best_genotypes.append(genotype_img)
    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'arch_param': model.arch_parameters(),
      'val_acc': valid_acc,
      'optimizer' : optimizer.state_dict(),
      }, False, args.save)
  
  with open('./genotypes.py', 'a') as f:
    f.write(args.geno_name+' = '+ str(best_genotypes[0])+'\n')
    f.write(args.geno_name+'_img'+' = '+ str(best_genotypes[1])+'\n')
def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, bin_op, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  for step, (input, target) in enumerate(train_queue):
    bin_op.binarization()

    n = input.size(0)

    input =input.cuda()
    target =target.cuda()

    input_search, target_search = next(iter(valid_queue))
    input_search = input_search.cuda()
    target_search = target_search.cuda()

    architect.step(input, target, input_search, target_search, lr, optimizer,epoch)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    bin_op.restore()
    bin_op.updateBinaryGradWeight()
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


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

      logits = model(input)
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

