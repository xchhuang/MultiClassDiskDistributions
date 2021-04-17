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
from asmcdd import ASMCDD
from trainer import Trainer
import utils
from PIL import Image
import sys
sys.path.append('../')


# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')    # seems cpu is faster
# command
# python main.py --config_filename=configs/zerg_rush.txt

# np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--config_filename', type=str, default='configs/zerg_rush.txt', help='config file name')
parser.add_argument('--refine', action='store_true')
parser.add_argument('--nSteps', type=int, default=50, help='number of bins for computing PCF')
parser.add_argument('--sigma', type=float, default=0.25, help='smoothness of Gaussian kernel sigma for computing PCF')
parser.add_argument('--domainLength', type=int, default=1, help='domain size: 1 for same domain and n for n times larger domain')

opt = parser.parse_args()

filenames = ['bird', 'bunny', 'dog']


def load_elements(categories):
    img_res = 512
    path_prefix = './elements'
    elements = []
    for filename in filenames:
        im = Image.open(path_prefix+'/'+filename+'.png').convert('RGBA')
        elements.append(im)
    # print(len(elements))
    # categories_img = []
    categories_elem = defaultdict(list)
    for k in categories.keys():     # assume number of classes is equal to number of elements type
        im = elements[k]
        im = im.resize((img_res, img_res), resample=Image.LANCZOS)
        im = np.asarray(im)     # [W, H, 3or4]
        sample_spheres = utils.getSamplesFromImage(im)
        # print(sample_spheres.shape)
        for e in categories[k]:
            cur_sample_sphere = sample_spheres.copy()
            ratio = sample_spheres[0, 2] / e[2]
            cur_sample_sphere[0, 2] = e[2]
            cur_sample_sphere[0, 0:2] = e[0:2]
            cur_sample_sphere[1:, 0:3] /= ratio
            cur_sample_sphere[1:, 0:2] += e[0:2]
            # print(cur_sample_sphere[1])
            categories_elem[k].append(cur_sample_sphere)
    # print(len(categories_elem[0]), categories_elem[0][0].shape)
    fig, ax = plt.subplots()
    for k in categories_elem.keys():
        out = categories_elem[k]
        out = np.array(out)
        for i in range(out.shape[0]):
            for j in range(1, out.shape[1]):
                # plt.scatter(out[:, 0], out[:, 1], s=5)
                circle = plt.Circle((out[i, j, 0], out[i, j, 1]), out[i, j, 2], color='r', fill=False)
                ax.add_artist(circle)
        plt.axis('equal')
        plt.xlim([-0.2, 1.2])
        plt.ylim([-0.2, 1.2])
    plt.show()


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
    radii = []
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
                categories[idx].append([x/10000.0, y/10000.0, r/10000.0])
                radii.append(r/10000.0)

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

        
        # Trainer(device, opt, categories, relations)
        load_elements(categories)

if __name__ == "__main__":
    main()

