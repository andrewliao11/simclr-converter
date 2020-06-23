import argparse
import torch
import numpy as np
from collections import Counter
from tqdm import tqdm
import ipdb


def str2bool(v):
    return v.lower() == 'true'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--norm', type=str2bool, default=False)
parser.add_argument('--n_workers', type=int, default=1)
args = parser.parse_args()


data = np.load("/scratch/gobi1/andrewliao/simclr/self_supervised_features.npz")
output = torch.tensor(data["output"])
target = data["target"]

if args.norm:
    print("Normalizing features")
    output = (output - output.mean(0)) / (output.std(0) + 1e-5)


output_g = output.cuda()
b, h = output.shape


sort_neighbor_arr = []
target_arr = []
dist_arr = []
splits = torch.split(output, 3, dim=0)
for split in tqdm(splits):
    s_b, _ = split.shape
    split = split.cuda()
    dist = output_g.reshape(1, b, h) - split.reshape(s_b, 1, h)
    dist = torch.norm(dist, 2, 2)

    dist = dist.cpu().numpy()
    for d in dist:
        sort_neighbor = np.argsort(d)[1:501]
        sort_target = np.array([target[i] for i in sort_neighbor])
        sort_dist = np.array([d[i] for i in sort_neighbor])

        sort_neighbor_arr.append(sort_neighbor)
        target_arr.append(sort_target)
        dist_arr.append(sort_dist)



sort_neighbor_arr = np.array(sort_neighbor_arr)
target_arr = np.array(target_arr)
dist_arr = np.array(dist_arr)

name = "knn_norm" if args.norm else "knn"
np.savez(name, sort_neighbor_arr=sort_neighbor_arr, target_arr=target_arr, dist_arr=dist_arr)
