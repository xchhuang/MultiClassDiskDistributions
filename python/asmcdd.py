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
from computeFunctions import PCF, PrettyPCF

# np.random.seed(0)


class ASMCDD(torch.nn.Module):
    def __init__(self, device, opt, categories, relations):
        super(ASMCDD, self).__init__()
        self.device = device
        self.opt = opt
        self.categories = categories
        self.relations = relations

        self.w = torch.nn.ParameterList()
        for k in categories.keys():
            self.w.append(torch.nn.Parameter(torch.from_numpy(np.array(categories[k])).float()))
        self.pcf_model = []
        self.pcf = []
        self.computeTarget()

    def computeTarget(self):
        n_classes = len(self.w)
        plt.figure(1)
        print(self.relations)
        for i in range(n_classes):
            category_i = i
            target_disks = self.categories[i]
            for j in range(len(self.relations[i])):
                category_j = self.relations[i][j]
                parent_disks = self.categories[category_j]
                # print(len(target_disk), len(parent_disk))

                target_disks = torch.from_numpy(np.array(target_disks)).double().to(self.device)
                parent_disks = torch.from_numpy(np.array(parent_disks)).double().to(self.device)
                same_category = False
                # print(category_i, category_j, 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * len(target_disks))))
                if category_i == category_j:
                    same_category = True
                # if not (category_i == 0 and category_j == 0):
                #     continue
                # print(category_j, category_i)
                cur_pcf_model = PCF(self.device, nbbins=50, sigma=0.25, npoints=len(target_disks), n_rmax=5).to(self.device)
                cur_pcf_mean, cur_pcf_min, cur_pcf_max = cur_pcf_model(target_disks, parent_disks, same_category=same_category, dimen=3, use_fnorm=True)

                # plt.figure(1)
                # plt.plot(cur_pcf_mean[:, 0].detach().cpu().numpy(), cur_pcf_mean[:, 1].detach().cpu().numpy())
                # plt.savefig(self.opt.output_folder + '/pcf_{:}_{:}'.format(category_i, category_j))
                # plt.clf()
                # pretty_pcf_model = PrettyPCF(self.device, nbbins=50, sigma=0.25, npoints=len(target_disks), n_rmax=5).to(self.device)
                # pretty_pcf = pretty_pcf_model(target_disks, parent_disks, same_category, dimen=3, use_fnorm=False)

                # plt.figure(1)
                plt.plot(cur_pcf_mean[:, 0].detach().cpu().numpy(), cur_pcf_mean[:, 1].detach().cpu().numpy())
                # plt.plot(pretty_pcf[:, 0].detach().cpu().numpy(), pretty_pcf[:, 1].detach().cpu().numpy())
                # plt.legend(['0_0', 'pcf_0'])
                # plt.savefig(self.opt.output_folder + '/pcf_{:}_{:}'.format(category_i, category_j))
                # plt.clf()
        plt.savefig(self.opt.output_folder + '/{:}_pcf'.format(self.opt.scene_name))
        plt.clf()

    def forward(self):
        for i in range(3):
            print(self.w[i].shape)
