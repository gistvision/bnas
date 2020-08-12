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

from model_imagenet import NetworkImageNet as Network
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
parser.add_argument('--learning_rate', type=float, default=0.05, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=64, help='num of init channels')
parser.add_argument('--layers', type=int, default=12, help='total number of layers')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--opt_level', type=str, default='O1')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)
parser.add_argument("--local_rank", default=0, type=int)
parser.add_argument('--resume', type=str, default = None,help='argument to resume')
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--num_skip', type=int, default = 3, help='number of conv to not binarize in the first parts of the network')
parser.add_argument('--class_size', type =int, default =1000, help='num of class size')
parser.add_argument('--model_config', type=str, default='bnas-d', help='model config to run inference')
args = parser.parse_args()


args.save = 'eval-{}'.format(args.save)

if args.local_rank == 0:
  if args.resume is None:
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  from tensorboardX import SummaryWriter
  writer_comment = args.save
  writer = SummaryWriter(logdir = args.save, comment=writer_comment)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if args.local_rank==0:
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

CLASSES = args.class_size

configs = {'bnas-d': [12,64, 'O0'], 'bnas-e':[12,68, 'O0'],'bnas-f':[15,68, 'O0'],'bnas-g':[11,74, 'O0'], 'bnas-h':[16,128, 'O0']}
args.layers = configs[args.model_config][0]
args.init_channels=configs[args.model_config][1]
args.opt_level = configs[args.model_config][2]

class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


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
  model = Network(args.init_channels, CLASSES, args.layers, genotype).cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )

  model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

  if args.distributed:
      model = DDP(model,delay_allreduce=True)
  bin_op = bin_utils.BinOp(model, args)
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()
  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  train_sampler = None
  val_sampler = None
  if args.distributed:
      train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
      val_sampler = torch.utils.data.distributed.DistributedSampler(valid_data)

  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=(train_sampler is None), pin_memory=True, num_workers=4, sampler=train_sampler)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4, sampler=val_sampler)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, eta_min = 0.0)
  best_acc_top1 = 0
  if args.resume:
        # Use a local scope to avoid dangling references
    def resume():
      if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location = lambda storage, loc: storage.cuda(args.gpu))
        args.start_epoch = checkpoint['epoch']
        best_acc_top1 = checkpoint['best_acc_top1']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        amp.load_state_dict(checkpoint['amp'])
        print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
      else:
        print("=> no checkpoint found at '{}'".format(args.resume))
        
    resume()

  for epoch in range(args.start_epoch, args.epochs):
    logging.info('epoch %d', epoch)
    if args.distributed:
      model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    else:
      model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(train_queue, model, criterion_smooth, optimizer, bin_op, scheduler, epoch)
    logging.info('train_acc %f', train_acc)
    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion, bin_op, epoch)
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)
    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True
    if args.local_rank == 0:
      utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'optimizer' : optimizer.state_dict(),
      'scheduler' : scheduler.state_dict(),
      'amp':amp.state_dict()
      }, is_best, args.save)
  
def train(train_queue, model, criterion, optimizer, bin_op, scheduler, epoch):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()
  iters = len(train_queue)
  for step, (input, target) in enumerate(train_queue):

    bin_op.binarization()
    target = target.cuda(async=True)
    input = input.cuda()
    scheduler.step(epoch + step/iters)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    with amp.scale_loss(loss, optimizer) as scaled_loss:
        scaled_loss.backward()
    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    bin_op.restore()
    bin_op.updateBinaryGradWeight()
    optimizer.step()

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
    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
    del loss, logits, input, target, reduced_loss, prec1, prec5
    gc.collect()
    torch.cuda.empty_cache()
  if args.local_rank == 0:
    writer.add_scalar('train/top1', top1.avg, epoch)
    writer.add_scalar('train/top5', top5.avg, epoch)
    writer.add_scalar('train/loss', objs.avg, epoch)
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, bin_op, epoch):
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

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  bin_op.restore()
  if args.local_rank == 0:
    writer.add_scalar('test/top1', top1.avg, epoch)
    writer.add_scalar('test/top5', top5.avg, epoch)
    writer.add_scalar('test/loss', reduced_loss, epoch)
  return top1.avg, top5.avg, objs.avg

if __name__ == '__main__':
  main() 
