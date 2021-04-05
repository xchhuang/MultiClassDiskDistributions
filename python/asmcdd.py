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
from collections import defaultdict

# np.random.seed(0)


def iterative_dfs(graph, start, path=[]):
    q = [start]
    while q:
        v = q.pop(0)
        if v not in path:
            path = path + [v]
            q = graph[v] + q

    return path


class ASMCDD(torch.nn.Module):
    def __init__(self, device, opt, categories, relations):
        super(ASMCDD, self).__init__()
        self.device = device
        self.opt = opt
        self.categories = categories
        self.relations = relations

        self.target = torch.nn.ParameterList()
        for k in categories.keys():
            self.target.append(torch.nn.Parameter(torch.from_numpy(np.array(categories[k])).float()))
        self.pcf_model = []
        # self.pcf = []
        self.computeTarget()
        self.initialize()

    def computeTarget(self):
        n_classes = len(self.target)
        plt.figure(1)
        print('relations:', self.relations)
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
                self.pcf_model.append(cur_pcf_model)
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
        # plt.ylim([0, 10])
        plt.savefig(self.opt.output_folder + '/{:}_pcf'.format(self.opt.scene_name))
        plt.clf()
        print('===> computeTarget Done.')

    def topologicalSort(self):
        graph = defaultdict(list)
        for k in self.relations:
            for v in self.relations[k]:
                if v != k:
                    graph[v].append(k)
        root_id = 0
        all_children = set()
        for i in graph:
            for j in graph[i]:
                all_children.add(j)
        all_children = list(all_children)
        for k in self.relations:    # find a root node id
            if k not in all_children:
                root_id = k
                break
        return graph, root_id

    def initialize(self, domainLength=1, e_delta=1e-4):
        # first find the topological oder
        graph, root_id = self.topologicalSort()
        # print(graph, root_id)
        topological_order = iterative_dfs(graph, root_id)
        # print('topological_order:', topological_order)

        n_factor = domainLength * domainLength
        diskfact = 1 / domainLength
        n_repeat = math.ceil(n_factor)
        # print(n_factor, diskfact, n_repeat)

        for i in topological_order:
            target_disks = self.categories[i]
            target_disks = torch.from_numpy(np.array(target_disks)).double().to(self.device)
            # print(target_disks.shape)
            output_disks_radii = torch.zeros(target_disks.shape[0]*n_repeat).double().to(self.device)
            output_disks_radii = target_disks[:, -1].repeat(n_repeat)
            idx = torch.randperm(output_disks_radii.shape[0])
            output_disks_radii = output_disks_radii[idx].view(output_disks_radii.size())
            output_disks_radii, _ = torch.sort(output_disks_radii, descending=True)
            # print(i, output_disks_radii.shape)

    def forward(self):
        for i in range(3):
            print(self.w[i].shape)
