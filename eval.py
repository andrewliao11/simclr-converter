"""
sgpu python eval.py /scratch/gobi1/datasets/imagenet -a resnet50-4x -b 32 -s train
"""
import argparse
import os
import random
import shutil
import json
import warnings
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from resnet_wider import resnet50x1, resnet50x2, resnet50x4

from my_dataset import ImageFolder
import ipdb


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', default='resnet50-1x')
parser.add_argument('-s', '--split', default='train')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int)
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

best_acc1 = 0


def main():
    args = parser.parse_args()

    # create model
    if args.arch == 'resnet50-1x':
        model = resnet50x1()
        sd = 'resnet50-1x.pth'
    elif args.arch == 'resnet50-2x':
        model = resnet50x2()
        sd = 'resnet50-2x.pth'
    elif args.arch == 'resnet50-4x':
        model = resnet50x4()
        sd = 'resnet50-4x.pth'
    else:
        raise NotImplementedError

    sd = torch.load(sd, map_location='cpu')
    model.load_state_dict(sd['state_dict'])
    model.fc = Identity()

    model = torch.nn.DataParallel(model).to('cuda')
    cudnn.benchmark = True

    # Data loading code
    valdir = os.path.join(args.data, args.split)

    # NOTICE, the original model do not have normalization
    val_loader = torch.utils.data.DataLoader(
        ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
        ]), return_path=True),
        batch_size=args.batch_size,
        #shuffle=False,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True)

    validate(val_loader, model, args)


def validate(val_loader, model, args):

    # switch to evaluate mode
    self_supervised_features = {}
    path_arr = []
    output_arr = []
    target_arr = []

    model.eval()

    with torch.no_grad():
        for i, (paths, images, target) in tqdm(enumerate(val_loader)):
            target = target.to('cuda')

            # compute output
            output = model(images)
            path_arr += [j for j in paths]
            output_arr += [j for j in output.cpu().numpy()]
            target_arr += [j for j in target.cpu().numpy()]

            if i > (1e6 // args.batch_size):
                break


    path_arr = np.array(path_arr)
    output_arr = np.array(output_arr)
    target_arr = np.array(target_arr)

    idx = list(val_loader.dataset.idx_to_class.keys())
    idx.sort()
    idx_to_class = np.array([val_loader.dataset.idx_to_class[i] for i in idx])
    np.savez("/scratch/gobi1/andrewliao/simclr/simclr-v1-{}".format(args.split),
             path=path_arr,
             output=output_arr,
             target=target_arr,
             idx_to_class=idx_to_class)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    main()
