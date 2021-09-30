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
from collections import defaultdict
# from asmcdd import ASMCDD
# from trainer import Trainer
from solver import Solver

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')  # seems cpu is faster
# command
# python main.py --config_filename=configs/zerg_rush.txt

# np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', type=str, default='configs/zerg_rush.txt', help='config file name')
parser.add_argument('--refine', action='store_true')
parser.add_argument('--nSteps', type=int, default=50, help='number of bins for computing PCF')
parser.add_argument('--sigma', type=float, default=0.25, help='smoothness of Gaussian kernel sigma for computing PCF')
parser.add_argument('--domainLength', type=int, default=1, help='domain size: 1 for same domain and n for n times larger domain')
parser.add_argument('--samples_per_element', type=int, default=0, help='samples per element')

opt = parser.parse_args()


#
# def plot_pts(title, pts):
#     plt.figure(1)
#     ax = plt.gca()
#     ax.cla()
#     plt.scatter(pts[:, 0], pts[:, 1], s=5, c='r')
#     ax.set_aspect('equal')
#     plt.savefig(opt.outd + '/' + title)
#     plt.clf()


def main():
    config_filename = opt.config_filename
    scene_name = config_filename.split('/')[-1].split('.')[0]
    opt.scene_name = scene_name
    output_folder = 'results/' + scene_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(output_folder, config_filename, scene_name)
    opt.output_folder = output_folder
    categories = defaultdict(list)
    relations = defaultdict(list)

    with open(config_filename) as f:
        lines = f.readlines()
        example_filename = lines[0].strip()
        num_dependancies = int(lines[1].strip())
        dependancies = lines[2:]

        # return
        id_map = {}  # init
        num_classes = -1  # init
        with open(example_filename) as e:
            ef = e.readlines()
            firstline = []
            for line in ef[0:1]:
                firstline = line.strip().split(' ')
                firstline = [int(x) for x in firstline]
                # print(firstline)
            num_classes = firstline[0]
            class_index = firstline[1:]
            for i in range(num_classes):
                id_map[class_index[i]] = i
            for line in ef[1:]:
                line = line.strip().split(' ')
                line = [int(x) for x in line]
                idx = id_map[line[0]]
                x, y, r = line[1], line[2], line[3]
                # print(idx)
                categories[idx].append([x / 10000.0, y / 10000.0, r / 10000.0])

        for k in range(num_classes):
            relations[k].append(k)
        for i in range(num_dependancies):
            dep = dependancies[i].strip().split(' ')
            dep = [int(x) for x in dep]
            id_a, id_b = dep
            relations[id_b].append(id_a)

        # print
        for k in categories.keys():
            print('#Disk of class {:} {:}, their parents {:}'.format(k, len(categories[k]), relations[k]))
            x = np.expand_dims(np.array(categories[k]), 1)
            # print(x.shape)
            categories[k] = x
            # print(categories[k])
        # print(categories, relations)
        # Trainer(device, opt, categories, relations)
        Solver(opt, categories, relations)


if __name__ == "__main__":
    main()
