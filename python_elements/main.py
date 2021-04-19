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
import renderer

sys.path.append('../')

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
parser.add_argument('--domainLength', type=int, default=1,
                    help='domain size: 1 for same domain and n for n times larger domain')
parser.add_argument('--samples_per_element', type=int, default=2, help='samples per element')
parser.add_argument('--n_iter', type=int, default=1, help='number of iterations in refinement step')

opt = parser.parse_args()

filenames = ['bird', 'bunny', 'dog']


def load_elements(categories):
    img_res = 512
    path_prefix = './elements'
    elements = []
    for filename in filenames:
        im = Image.open(path_prefix + '/' + filename + '.png').convert('RGB')
        elements.append(im)
    # print(len(elements))
    # categories_img = []
    categories_elem = defaultdict(list)
    categories_radii_ratio = defaultdict(list)
    for k in categories.keys():  # assume number of classes is equal to number of elements type
        im = elements[k]
        im = im.resize((img_res, img_res), resample=Image.LANCZOS)
        im = np.asarray(im)  # [W, H, 3or4]
        sample_spheres = utils.getSamplesFromImage(im, opt.samples_per_element)
        # print(sample_spheres.shape)
        for e in categories[k]:
            rotate_factor = 0   # np.random.rand() * 2 * np.pi    # random angle to rotate

            cur_sample_sphere = sample_spheres.copy()
            ratio = sample_spheres[0, 2] / e[2]
            categories_radii_ratio[k].append(ratio)
            # print('ratio:', sample_spheres[0, 2], e[2])
            cur_sample_sphere[0] = e    # center, the same
            # cur_sample_sphere[0, 0:2] = e[0:2]
            cur_sample_sphere[1:, 0:3] /= ratio     # sample_spheres need to be scaled and rotated
            tmp_copy = cur_sample_sphere[1:, 0:2].copy()

            cur_sample_sphere[1:, 0:1] = np.cos(rotate_factor) * tmp_copy[:, 0:1] - np.sin(rotate_factor) * tmp_copy[:,
                                                                                                            1:2]
            cur_sample_sphere[1:, 1:2] = np.sin(rotate_factor) * tmp_copy[:, 0:1] + np.cos(rotate_factor) * tmp_copy[:,
                                                                                                            1:2]

            cur_sample_sphere[1:, 0:2] += e[0:2]    # moved to e's positions
            # print(cur_sample_sphere[1])
            categories_elem[k].append(cur_sample_sphere)
    # print(len(categories_elem[0]), categories_elem[0][0].shape)
    # utils.plot_elements(categories_elem.keys(), categories_elem, opt.output_folder+'/target_elements')
    return categories_elem, categories_radii_ratio, elements


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
                categories[idx].append([x / 10000.0, y / 10000.0, r / 10000.0])
                radii.append(r / 10000.0)

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

        categories_elem, categories_radii_ratio, elements = load_elements(categories)
        # renderer.render(categories_elem, categories_radii_ratio, elements)
        Trainer(device, opt, categories, categories_elem, categories_radii_ratio, elements, relations)


if __name__ == "__main__":
    main()
