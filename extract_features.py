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
        ]), return_path=True, split="1/3"),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)


    validate(val_loader, model, args)


def validate(val_loader, model, args):

    # switch to evaluate mode
    path_arr = []
    output_arr = []
    target_arr = []

    model.eval()

    with torch.no_grad():
        for i, (paths, images, targets) in tqdm(enumerate(val_loader)):

            # compute output
            output = model(images)
            path_arr.extend(paths)
            output_arr.append(output.cpu().numpy())
            target_arr.append(targets.cpu().numpy())
            if i % 100 == 0:
                print("{}/{}".format(i, len(val_loader)))


    path_arr = np.array(path_arr)
    output_arr = np.concatenate(output_arr, 0)
    target_arr = np.concatenate(target_arr, 0)

    idx = list(val_loader.dataset.idx_to_class.keys())
    idx.sort()
    idx_to_class = np.array([val_loader.dataset.idx_to_class[i] for i in idx])
    np.savez("/scratch/gobi2/andrewliao/simclr/simclr-v1-{}-1/3".format(args.split),
             path=path_arr,
             output=output_arr,
             target=target_arr,
             idx_to_class=idx_to_class)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


if __name__ == '__main__':
    main()
