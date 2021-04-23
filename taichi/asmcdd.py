import numpy as np
import matplotlib.pyplot as plt
import argparse
from time import time
import torch
import math
from tqdm import tqdm
import os
import glob
from computeFunctions import PCF, PrettyPCF, PCF_utk
from computeFunctions_ti import PCF_ti

from collections import defaultdict
from utils import Contribution
# from computeFunctions import compute_contribution, compute_error
import utils
# import copy
# from graphlib import TopologicalSorter
# from refiner import Refiner
import taichi as ti

ti.init(arch=ti.cpu)


# def get_weight(d_test, cur_pcf_model, diskfact):
#     weight = cur_pcf_model.perimeter_weight(d_test[0], d_test[1], diskfact)
#     return weight


# def get_weights(disks, cur_pcf_model, diskfact):
#     weights = []
#     for p in disks:
#         weight = get_weight(p, cur_pcf_model, diskfact)
#         weights.append(weight)
#     # weights = torch.stack(weights, 0)
#     # print('get_weights:', weights.shape)
#     return weights


@ti.data_oriented
class ASMCDD:
    def __init__(self, device, opt, categories, relations):
        super(ASMCDD, self).__init__()
        self.device = device
        self.opt = opt
        self.categories = categories
        self.relations = relations
        self.nSteps = opt.nSteps
        self.sigma = opt.sigma

        self.target_pcf_model = defaultdict(dict)   # PCF model, record area, radii, max, etc.
        self.output_pcf_model = defaultdict(dict)
        self.target_pcf = defaultdict(list)     # PCFs, record pcfs of nSteps bins
        self.output_disks_radii = defaultdict(list)     # sort radii for synthesis usage
        self.current_pcf = defaultdict(list)    # tmp
        self.contributions = defaultdict(Contribution)  # tmp
        self.weights = defaultdict(list)    # tmp

        self.count_disks = defaultdict(int)     # record number of disks already added
        self.d_test_ti = ti.field(dtype=ti.f32, shape=3)    # tmp
        self.test_pcf = Contribution(self.nSteps)   # tmp
        self.tmp_weight = ti.field(dtype=ti.f32, shape=self.nSteps)     # tmp

        self.domainLength = opt.domainLength
        self.n_factor = self.domainLength * self.domainLength
        self.diskfact = 1
        self.cur_id = 0
        self.cur_parent_id = 0
        self.n_repeat = math.ceil(self.n_factor)

        # max_num_disks = -np.inf
        # for i in categories.keys():
        #     cur_num = len(categories[i])
        #     if max_num_disks < cur_num:
        #         max_num_disks = cur_num

        # define taichi fields in PCF_ti
        for i in categories.keys():
            self.output_disks_radii[i] = ti.field(dtype=ti.f32, shape=len(self.categories[i])*self.n_repeat)
            self.current_pcf[i] = ti.field(dtype=ti.f32, shape=self.nSteps)
            self.contributions[i] = Contribution(self.nSteps)
            # self.weights[i] = ti.Vector.field(max_num_disks, dtype=ti.f32, shape=self.nSteps)
            for j in self.relations[i]:
                Nk = len(self.categories[i])
                same_category = False
                if i == j:
                    same_category = True
                self.target_pcf_model[i][j] = PCF_ti(disks=self.categories[i], disks_parent=self.categories[j],
                                                 same_category=same_category, nbbins=self.nSteps, sigma=self.sigma,
                                                 npoints=Nk, n_rmax=5, domainLength=1)
                print('relation ids:', i, j)
                self.output_pcf_model[i][j] = PCF_ti(disks=self.categories[i], disks_parent=self.categories[j],
                                                     same_category=same_category, nbbins=self.nSteps, sigma=self.sigma,
                                                     npoints=Nk, n_rmax=5, domainLength=self.opt.domainLength)

        # initialize taichi fields inside pcf_model_ti
        for i in categories.keys():
            cur_tar_disks = np.array(self.categories[i])
            radii = cur_tar_disks[:, -1].repeat(self.n_repeat) / self.domainLength
            radii = np.sort(radii)[::-1]
            self.output_disks_radii[i].from_numpy(radii)
            self.current_pcf[i].from_numpy(np.zeros(self.nSteps))
            self.contributions[i].initialize()
            # self.weights[i].from_numpy(np.zeros((self.nSteps)))
            self.count_disks[i] = 0
            for j in self.relations[i]:
                self.target_pcf_model[i][j].initialize()
                self.output_pcf_model[i][j].initialize()

        start = time()
        self.computeTarget_ti()
        end = time()
        print('Time of computing target PCFs(taichi): {:.4f}'.format(end - start))
        # self.plot_pretty_pcf(self.target, self.target, self.relations)

        self.initialize(domainLength=opt.domainLength)
        #
        #
        # utils.plot_disks(self.topological_order, self.target, self.opt.output_folder + '/target')
        # self.plot_pretty_pcf(self.target, self.outputs, self.relations)

        # utils.plot_disks(self.topological_order, self.outputs, self.opt.output_folder + '/output')

    @ti.kernel
    def get_weight(self):
        model = self.output_pcf_model[self.cur_id][self.cur_id]
        for k in range(self.nSteps):
            self.test_pcf.weights[k] = model.perimeter_weight_ti(self.d_test_ti[0], self.d_test_ti[1], model.rs_ti[k])

    # @ti.kernel
    # def get_weights_helper(self):

    # def get_weights(self, disks, cur_pcf_model, diskfact):
    @ti.kernel
    def get_weights(self):
        N = self.count_disks[self.cur_parent_id]     # global dict to count number of synthesized disks
        model = self.output_pcf_model[self.cur_parent_id][self.cur_parent_id]

        print('before inside:', len(self.weights[self.cur_parent_id]))
        for _ in range(1):
            for ii in range(N):
                print(ii)
                p = model.disks_ti[ii]
                for k in range(self.nSteps):
                    self.tmp_weight[k] = model.perimeter_weight_ti(p[0], p[1], model.rs_ti[k])
                self.weights[self.cur_parent_id].append(self.tmp_weight)
        print('inside:', len(self.weights[self.cur_parent_id]))

    @ti.kernel
    def compute_contribution(self):
        pass

    def computeTarget_ti(self):
        # n_classes = len(self.target)
        # print('relations:', self.relations)
        for i in self.categories.keys():
            # target_disks = self.target[i]
            for j in self.relations[i]:
                self.target_pcf_model[i][j].forward()
                # self.output_pcf_model[i][j].forward()

                cur_pcf_mean = self.target_pcf_model[i][j].pcf_mean.to_numpy()
                # cur_pcf_mean2 = self.output_pcf_model[i][j].pcf_mean.to_numpy()

                plt.plot(cur_pcf_mean[:, 0], cur_pcf_mean[:, 1])
                # print(cur_pcf_mean[:, 1])
                plt.savefig(
                    self.opt.output_folder + '/{:}_pcf_{:}_{:}'.format(self.opt.scene_name, i, j))
                plt.clf()
                return
        print('===> computeTarget Done.')

    def plot_pretty_pcf(self, exemplar, output, relations):
        n_classes = len(exemplar)
        for i in range(n_classes):
            category_i = i
            exemplar_disks = exemplar[category_i]
            output_disks = output[category_i]
            for j in range(len(relations[i])):
                plt.figure(1)
                category_j = relations[i][j]
                exemplar_parent_disks = exemplar[category_j]
                output_parent_disks = output[category_j]
                same_category = False
                if category_i == category_j:
                    same_category = True
                pretty_pcf_model = PrettyPCF(self.device, nbbins=self.opt.nSteps, sigma=0.25,
                                             npoints=len(exemplar_disks), n_rmax=5).to(self.device)

                exemplar_pcf_mean = pretty_pcf_model(exemplar_disks, exemplar_parent_disks, same_category)
                output_pcf_mean = pretty_pcf_model(output_disks, output_parent_disks, same_category)

                # self.pcf_model[category_i][category_j] = pretty_pcf_model
                # self.target_pcfs[category_i][category_j] = exemplar_pcf_mean

                # plt.figure(1)
                plt.plot(exemplar_pcf_mean[:, 0].detach().cpu().numpy(), exemplar_pcf_mean[:, 1].detach().cpu().numpy(),
                         'r-')
                plt.plot(output_pcf_mean[:, 0].detach().cpu().numpy(), output_pcf_mean[:, 1].detach().cpu().numpy(),
                         'g-')
                plt.legend(['exemplar', 'output'])
                plt.savefig(
                    self.opt.output_folder + '/{:}_pcf_{:}_{:}'.format(self.opt.scene_name, category_i, category_j))
                plt.clf()
        print('===> Plot Pretty PCF, Done.')

    def initialize(self, domainLength=1, e_delta=1e-4):
        # first find the topological oder
        graph, topological_order = utils.topologicalSort(len(self.categories.keys()), self.relations)
        self.topological_order = topological_order
        print('topological_order:', topological_order)

        for i in topological_order:

            # TODO: those params should be defined inside the loop
            e_0 = 0
            max_fails = 1000
            fails = 0
            n_accepted = 0

            # target_disks = self.categories[i]
            target_disks = self.target_pcf_model[i][i].disks_ti
            output_radii = self.output_disks_radii[i]
            # print(target_disks.shape, output_radii)

            self.cur_id = i     # make it global

            for relation in self.relations[i]:
                self.current_pcf[relation].from_numpy(np.zeros(self.nSteps))
                self.cur_parent_id = relation
                # print('before:', len(self.weights[self.cur_parent_id]))
                self.get_weights()
                self.contributions[relation].initialize()

            """
            Initialization step:
            """
            n_accepted = self.count_disks[i]
            print('n_accepted:', n_accepted, self.output_disks_radii[i].shape[0])
            start_time = time()
            while n_accepted < output_radii.shape[0]:
                # print(
                #     '===> Before Gird Search, n_accepted: {:}/{:}'.format(n_accepted + 1, output_disks_radii.shape[0]))
                #
                rejected = False
                e = e_0 + e_delta * fails
                rx = np.random.rand()  # * domainLength
                ry = np.random.rand()  # * domainLength
                # if i == 0:
                #     min_xy = 0.09
                #     rx = min_xy + np.random.rand() * (domainLength - min_xy * 2)  # my version
                #     ry = min_xy + np.random.rand() * (domainLength - min_xy * 2)
                # print(rx, ry)
                d_test = np.array([rx, ry, output_radii[n_accepted]])
                self.d_test_ti.from_numpy(d_test)

                for relation in self.relations[i]:
                    self.test_pcf.initialize()
                    if self.count_disks[i] != 0 or i != relation:
                        same_category = False
                        if i == relation:
                            same_category = True
                        if same_category:
                            target_size = 2 * output_radii.shape[0] * output_radii.shape[0]
                        else:
                            target_size = 2 * output_radii.shape[0] * self.count_disks[relation]


                        # test_pcf = compute_contribution(self.device, self.nSteps, d_test, others_rel_torch,
                        #                                 weights_rel_torch, cur_pcf_model, same_category,
                        #                                 target_size, diskfact)
                        # if i == 1 and relation == 0:
                        #     print(n_accepted, test_pcf.contribution)
                        ce = compute_error(test_pcf, current_pcf[relation], self.target_pcfs[i][relation])

                        if e < ce:
                            rejected = True
                            break
                        else:
                            pass
                            # print(n_accepted, ce, e)

                    else:
                        # print('here2')
                        self.get_weight()

                    self.contributions[relation] = self.test_pcf


                if rejected:
                    fails += 1
                else:
                    print('===> Before Gird Search, n_accepted: {:}/{:}'.format(n_accepted + 1, output_radii.shape[0]))

                    # others[i].append(d_test)
                    self.output_pcf_model[i][i].disks_ti[self.count_disks[i]] = d_test
                    self.count_disks[i] += 1
                    # print(self.count_disks[i])
                    # print(n_accepted, self.cur_parent_id)
                    # print(len(self.weights[relation]))
                    fails = 0
                    for relation in self.relations[i]:
                        current = self.current_pcf[relation]
                        contrib = self.contributions[relation]
                        if relation == i:
                            pass
                            # TODO: make self.weights a taichi field, instead of a list
                            # self.weights[relation].append(contrib.weights)
                            # print(len(self.weights[relation]))
                        # print(self.current_pcf[relation].shape, contrib.contribution.shape)
                        for k in range(self.nSteps):
                            self.current_pcf[relation][k] += contrib.contribution[k]
                    n_accepted += 1

                if fails > max_fails:
                    # exceed fail threshold, switch to a parallel grid search
                    N_I = N_J = 80 * int(domainLength)  # 100
                    minError_contrib = defaultdict(Contribution)
                    for relation in self.relations[i]:
                        minError_contrib[relation] = Contribution(self.device, self.nSteps)

                    while n_accepted < output_disks_radii.shape[0]:
                        print('===> Doing Grid Search: {:}/{:}'.format(n_accepted + 1, output_disks_radii.shape[0]))
                        minError_val = np.inf
                        minError_i = 0
                        minError_j = 0
                        # minError_contrib = defaultdict(Contribution)
                        # for relation in self.relations[i]:
                        #     minError_contrib[relation] = Contribution(self.device, self.nSteps)
                        currentError = 0
                        for ni in range(1, N_I):
                            for nj in range(1, N_J):
                                gx = 1 / N_I * ni
                                gy = 1 / N_J * nj
                                cell_test = [gx, gy, output_disks_radii[n_accepted]]
                                # if min_x > gx > max_x or min_y > gy > max_y:
                                #     continue
                                cell_test = torch.from_numpy(np.array(cell_test)).float().to(self.device)

                                currentError = 0
                                current_contrib = defaultdict(Contribution)
                                for relation in self.relations[i]:
                                    current_contrib[relation] = Contribution(self.device, self.nSteps)

                                for relation in self.relations[i]:
                                    # test_pcf = Contribution(self.device, self.nSteps)
                                    same_category = False
                                    if relation == i:
                                        same_category = True
                                    if same_category:
                                        target_size = output_disks_radii.shape[0] * output_disks_radii.shape[0]
                                    else:
                                        target_size = output_disks_radii.shape[0] * len(others[relation])

                                    others_rel_torch = torch.stack(others[relation], 0)  # torch.Tensor: [N, 3]
                                    weights_rel_torch = torch.stack(weights[relation], 0)
                                    # print(cell_test.shape, others_rel_torch.shape, weights_rel_torch.shape, same_category)
                                    test_pcf = compute_contribution(self.device, self.nSteps, cell_test,
                                                                    others_rel_torch,
                                                                    weights_rel_torch, cur_pcf_model, same_category,
                                                                    target_size, diskfact)
                                    ce = compute_error(test_pcf, current_pcf[relation],
                                                       self.target_pcfs[i][relation])
                                    currentError = max(ce, currentError)
                                    current_contrib[relation] = test_pcf

                                # print(currentError)
                                if currentError < minError_val:
                                    minError_val = currentError
                                    minError_i = ni
                                    minError_j = nj
                                    for relation in self.relations[i]:
                                        minError_contrib[relation].pcf = current_contrib[relation].pcf.clone()
                                        minError_contrib[relation].weights = current_contrib[relation].weights.clone()
                                        minError_contrib[relation].contribution = current_contrib[
                                            relation].contribution.clone()

                        # finally outside grid search, then add random x, y offsets within a grid
                        x_offset = (np.random.rand() * 1 - 1 / 2) / (N_I * 10)
                        y_offset = (np.random.rand() * 1 - 1 / 2) / (N_J * 10)
                        cell_test = [1 / N_I * minError_i + x_offset,
                                     1 / N_J * minError_j + y_offset,
                                     output_disks_radii[n_accepted]]
                        cell_test = torch.from_numpy(np.array(cell_test)).float().to(self.device)
                        others[i].append(cell_test)

                        for relation in self.relations[i]:
                            current = current_pcf[relation]
                            contrib = minError_contrib[relation]
                            if relation == i:
                                weights[relation].append(contrib.weights)
                            current_pcf[relation] += contrib.contribution
                        n_accepted += 1

            class_done.append(i)  # class i initialization done
            end_time = time()
            print('===> Initialization time: {:.4f}'.format(end_time - start_time))

            """
            TODO: Refinement step:
            """
            # category_i = i
            # current_output = torch.stack(others[category_i], 0)
            # model_refine = Refiner(self.device, self.opt, current_output).to(self.device)
            # optimizer = torch.optim.Adam(model_refine.parameters(), lr=1e-4)
            # n_iter = 10
            # start_time = time()
            # for itr in range(1, n_iter + 1):
            #     cur_all_outputs = []
            #     if itr == 1:
            #         for k in class_done:
            #             cur_all_outputs.append(torch.stack(others[k], 0))
            #         utils.plot_disks(class_done, cur_all_outputs, self.opt.output_folder + '/init_{:}'.format(i))
            # 
            #     optimizer.zero_grad()
            #     out = model_refine()
            #     loss = torch.zeros(1).to(self.device)
            #     for j in range(len(self.relations[i])):
            #         category_j = self.relations[i][j]
            #         current_output_parent = torch.stack(others[category_j], 0)
            #         # print(current_output.shape, current_output_parent.shape)
            #         same_category = False
            #         if category_i == category_j:
            #             same_category = True
            #             current_output_parent = out.clone()  # parent changed as well
            # 
            #         # cur_pcf_model = self.pcf_model[category_i][category_j]
            #         exemplar_pcf_mean = self.target_pcfs[category_i][category_j][:, 0:2].detach()
            # 
            #         cur_pcf_model = PCF(self.device, nbbins=self.nSteps, sigma=self.sigma, npoints=len(others[i]),
            #                             n_rmax=5).to(self.device)
            # 
            #         output_pcf_mean, _, _ = cur_pcf_model(out*1, current_output_parent*1, same_category=same_category,
            #                                               dimen=3, use_fnorm=True, domainLength=domainLength)
            #         cur_loss = torch.nn.functional.mse_loss(output_pcf_mean[:, 1], exemplar_pcf_mean[:, 1])
            #         # print(cur_loss)
            #         loss += cur_loss
            # 
            #         save_pcf_prefix = 'init'
            #         if itr == 1:
            #             save_pcf_prefix = 'init'
            #         else:
            #             save_pcf_prefix = 'opt'
            #         if itr == 1 or itr % 10 == 0 or itr == n_iter:
            #             plt.figure(1)
            #             plt.plot(exemplar_pcf_mean.detach().cpu().numpy()[:, 0],
            #                      exemplar_pcf_mean.detach().cpu().numpy()[:, 1], 'r-')
            #             plt.plot(output_pcf_mean.detach().cpu().numpy()[:, 0],
            #                      output_pcf_mean.detach().cpu().numpy()[:, 1], 'g-')
            #             plt.title('Category {:}, Parent {:}'.format(category_i, category_j))
            #             plt.legend(['exemplar', 'output'])
            #             plt.savefig(self.opt.output_folder + '/'+save_pcf_prefix+'_pcf_{:}_{:}'.format(category_i, category_j))
            #             plt.clf()
            # 
            #     loss.backward()
            #     optimizer.step()
            #     if itr % 10 == 0 or itr == n_iter:
            #         print(itr, loss.item())
            #         out = model_refine().detach()
            #         others[i] = list(torch.unbind(out))
            #         for k in class_done:
            #             cur_all_outputs.append(torch.stack(others[k], 0))
            #         utils.plot_disks(class_done, cur_all_outputs, self.opt.output_folder + '/refined_{:}'.format(i))
            # 
            # end_time = time()
            # print('Refinement time: {:.4f}'.format(end_time - start_time))

        """
        Final output:
        """
        for k in topological_order:
            # if k > 1:
            #     break
            self.outputs.append(torch.stack(others[k], 0))

        # self.plot_pretty_pcf(self.target, self.outputs, self.relations)
