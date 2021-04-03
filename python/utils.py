import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import time
import torch
import math
from tqdm import tqdm
import datetime
import os
import glob

from trainers.trainer import Trainer

# np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--file', type=str, default='../data/c10.dat', help='point set file')
parser.add_argument('--lr', type=float, default='1e-4', help='learning rate')
parser.add_argument('--n_iter', type=int, default=200, help='number of iterations')

opt = parser.parse_args()
time_stamp = datetime.datetime.now().strftime("%y_%m_%d_%H_%M")
opt.outd = 'results/test'
if not os.path.exists(opt.outd):
    os.makedirs(opt.outd)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def plot_pts(title, pts):
    plt.figure(1)
    ax = plt.gca()
    ax.cla()
    plt.scatter(pts[:, 0], pts[:, 1], s=5, c='r')
    ax.set_aspect('equal')
    plt.savefig(opt.outd + '/' + title)
    plt.clf()


def main():
    Trainer(device=device)


if __name__ == "__main__":
    main()
