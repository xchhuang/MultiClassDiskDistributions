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
from computeFunctions import PCF, PrettyPCF, PCF_utk
from collections import defaultdict
from utils import Contribution
from computeFunctions import compute_contribution, compute_error
import utils
import copy
from graphlib import TopologicalSorter
from refiner import Refiner


# np.random.seed(0)


def get_weight(d_test, cur_pcf_model, diskfact):
    """
    :param d_test: [samples_per_element+1, 3]
    :param cur_pcf_model:
    :param diskfact: 1
    :return:
    """
    weight = cur_pcf_model.perimeter_weight(d_test[0, 0], d_test[0, 1], diskfact)
    return weight


def get_weights(disks, cur_pcf_model, diskfact):
    weights = []
    for p in disks:
        # print('get_weights:', p.shape)
        weight = get_weight(p, cur_pcf_model, diskfact)
        weights.append(weight)
    # weights = torch.stack(weights, 0)
    # print('get_weights:', weights.shape)
    return weights


class ASMCDD(torch.nn.Module):
    def __init__(self, device, opt, categories, categories_elem, relations):
        super(ASMCDD, self).__init__()
        self.device = device
        self.opt = opt
        self.categories = categories
        self.categories_elem = categories_elem
        self.relations = relations
        self.nSteps = opt.nSteps
        self.sigma = opt.sigma
        self.target = torch.nn.ParameterList()
        for k in categories.keys():
            x = torch.from_numpy(np.array(categories_elem[k])).float()[:, :, :]
            # print(self.opt.samples_per_element, x.shape)
            # print('read target:', x.shape)
            self.target.append(torch.nn.Parameter(x))

        self.pcf_model = defaultdict(dict)
        self.target_pcfs = defaultdict(dict)
        self.computeTarget()
        self.outputs = []

        self.plot_pretty_pcf(self.target, self.target, self.relations)
        utils.plot_elements(categories.keys(), self.target, self.opt.output_folder + '/target_elements')
        self.initialize(domainLength=opt.domainLength)

        self.plot_pretty_pcf(self.target, self.outputs, self.relations)
        utils.plot_elements(self.topological_order, self.outputs, self.opt.output_folder + '/output_elements')

    def computeTarget(self):
        n_classes = len(self.target)
        # plt.figure(1)
        print('relations:', self.relations)
        for i in range(n_classes):
            category_i = i
            target_disks = self.target[i]
            for j in range(len(self.relations[i])):
                plt.figure(1)
                category_j = self.relations[i][j]
                parent_disks = self.target[category_j]
                same_category = False
                if category_i == category_j:
                    same_category = True

                cur_pcf_model = PCF(self.device, nbbins=self.nSteps, sigma=self.sigma, npoints=len(target_disks),
                                    n_rmax=5).to(self.device)
                print('Target Info, id: {:}, parent: {:}, rmax: {:.4f}'.format(category_i, category_j,
                                                                               cur_pcf_model.rmax))
                cur_pcf_mean, cur_pcf_min, cur_pcf_max = cur_pcf_model(target_disks, parent_disks,
                                                                       same_category=same_category, dimen=3,
                                                                       use_fnorm=True)
                # print(cur_pcf_max)
                self.pcf_model[category_i][category_j] = cur_pcf_model
                self.target_pcfs[category_i][category_j] = torch.cat([cur_pcf_mean, cur_pcf_min.unsqueeze(1),
                                                                      cur_pcf_max.unsqueeze(1)], 1)

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
                plt.savefig(
                    self.opt.output_folder + '/{:}_pcf_{:}_{:}'.format(self.opt.scene_name, category_i, category_j))
                plt.clf()
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
                # cur_pcf_model = PCF(self.device, nbbins=self.opt.nSteps, sigma=0.25, npoints=len(exemplar_disks),
                #                     n_rmax=5).to(self.device)
                # exemplar_pcf_mean, exemplar_pcf_min, exemplar_pcf_max = cur_pcf_model(exemplar_disks,
                #                                                                       exemplar_parent_disks,
                #                                                                       same_category=same_category,
                #                                                                       dimen=3,
                #                                                                       use_fnorm=True)
                #
                # output_pcf_mean, output_pcf_min, output_pcf_max = cur_pcf_model(output_disks,
                #                                                                 output_parent_disks,
                #                                                                 same_category=same_category,
                #                                                                 dimen=3,
                #                                                                 use_fnorm=True)
                #
                # self.pcf_model[category_i][category_j] = cur_pcf_model
                # self.target_pcfs[category_i][category_j] = torch.cat([exemplar_pcf_mean, exemplar_pcf_min.unsqueeze(1),
                #                                                       exemplar_pcf_max.unsqueeze(1)], 1)
                # print('pretty:', len(exemplar_disks))
                pretty_pcf_model = PrettyPCF(self.device, nbbins=self.opt.nSteps, sigma=0.25,
                                             npoints=len(exemplar_disks), n_rmax=5).to(self.device)

                exemplar_pcf_mean = pretty_pcf_model(exemplar_disks[:, 0, :], exemplar_parent_disks[:, 0, :], same_category)
                output_pcf_mean = pretty_pcf_model(output_disks[:, 0, :], output_parent_disks[:, 0, :], same_category)

                # self.pcf_model[category_i][category_j] = pretty_pcf_model
                # self.target_pcfs[category_i][category_j] = exemplar_pcf_mean

                # plt.figure(1)
                plt.plot(exemplar_pcf_mean[:, 0].detach().cpu().numpy(), exemplar_pcf_mean[:, 1].detach().cpu().numpy(),
                         'r-')
                plt.plot(output_pcf_mean[:, 0].detach().cpu().numpy(), output_pcf_mean[:, 1].detach().cpu().numpy(),
                         'g-')
                plt.legend(['exemplar', 'output'])
                plt.savefig(
                    self.opt.output_folder + '/{:}_point_pcf_{:}_{:}'.format(self.opt.scene_name, category_i, category_j))
                plt.clf()
        print('===> Plot Pretty PCF, Done.')

    def initialize(self, domainLength=1, e_delta=1e-4):
        # first find the topological oder
        graph, topological_order = utils.topologicalSort(len(self.target), self.relations)
        self.topological_order = topological_order
        print('topological_order:', topological_order)

        n_factor = domainLength * domainLength
        # diskfact = 1 / domainLength
        diskfact = 1
        n_repeat = math.ceil(n_factor)
        # print(n_factor, diskfact, n_repeat)
        # disks = []
        others = defaultdict(list)  # existing generated disks
        for i in topological_order:
            others[i] = []
        # print(topological_order)

        class_done = []

        for i in topological_order:
            # if i > 1:
            #     break
            # TODO: those params should be defined inside the loop
            e_0 = 0
            max_fails = 1000
            fails = 0
            n_accepted = 0

            target_disks = self.target[i]
            # print(target_disks.shape)
            output_disks_radii = torch.zeros(target_disks.shape[0] * n_repeat).float().to(self.device)
            output_disks_radii = target_disks[:, 0, -1].repeat(n_repeat)
            # print(output_disks_radii.shape)
            _, idx = torch.sort(output_disks_radii, descending=True)
            # print(idx.shape)
            # output_disks_radii = output_disks_radii[idx]

            output_disks_radii = target_disks.repeat(n_repeat, 1, 1)[idx]
            # print(output_disks_radii.shape)
            # print(output_disks_radii[:, 0, -1])
            output_disks_radii[:, :, -1] /= domainLength
            c = output_disks_radii[:, 0, 0:2]
            output_disks_radii[:, :, 0:2] -= c.unsqueeze(1).repeat(1, self.opt.samples_per_element+1, 1)
            # print(c.shape, output_disks_radii[:, :, 0:2].shape)
            # print(output_disks_radii[:, 0, -1])
            out = defaultdict(list)
            out[0] = output_disks_radii
            utils.plot_elements([0], out, self.opt.output_folder + '/test_{:}'.format(i))

            # init
            # print(i, self.relations[i])
            current_pcf = defaultdict(list)
            contributions = defaultdict(Contribution)
            weights = defaultdict(list)
            cur_pcf_model = PCF(self.device, nbbins=self.nSteps, sigma=self.sigma, npoints=len(target_disks), n_rmax=5).to(
                self.device)

            for relation in self.relations[i]:
                current_pcf[relation] = torch.zeros(self.nSteps).float().to(self.device)
                # print(current_pcf.shape)
                others_disks = others[relation]
                # print(i, relation, others_disks)
                if len(others_disks) != 0:
                    others_disks = torch.stack(others_disks, 0)
                else:
                    others_disks = torch.from_numpy(np.array(others_disks)).float().to(self.device)
                # print(len(others_disks))
                current_weight = get_weights(others_disks, cur_pcf_model, diskfact)  # TODO: check, might be empty
                weights[relation] = current_weight
                contributions[relation] = Contribution(self.device, self.nSteps)

            """
            Initialization step:
            """
            start_time = time()
            while n_accepted < output_disks_radii.shape[0]:
                # print(
                #     '===> Before Gird Search, n_accepted: {:}/{:}'.format(n_accepted + 1, output_disks_radii.shape[0]))
                #
                rejected = False
                e = e_0 + e_delta * fails
                # if i == 0:
                # current_radius = output_disks_radii[n_accepted]

                # if i == 0:
                #     print(current_radius, (domainLength - current_radius))
                # else:
                rx = np.random.rand()  # * domainLength
                ry = np.random.rand()  # * domainLength
                # if i == 0:
                #     min_xy = 0.09
                #     rx = min_xy + np.random.rand() * (domainLength - min_xy * 2)  # my version
                #     ry = min_xy + np.random.rand() * (domainLength - min_xy * 2)
                # print(rx, ry)
                # print(output_disks_radii[n_accepted].shape)
                d_test = output_disks_radii[n_accepted].clone()
                d_test[:, 0] += rx
                d_test[:, 1] += ry
                # d_test = [rx, ry, output_disks_radii[n_accepted]]
                # d_test = torch.from_numpy(np.array(d_test)).float().to(self.device)  # d_test: torch.Tensor: (3,)

                for relation in self.relations[i]:
                    # if i == 0:
                    #     d_test = torch.from_numpy(np.array(predefined_id_pts[i][n_accepted])).float().to(self.device)
                    #     rejected = False
                    #     break
                    # if i == 1:
                    #     d_test = torch.from_numpy(np.array(predefined_id_pts[i][n_accepted])).float().to(self.device)

                    test_pcf = Contribution(self.device, self.nSteps)
                    if len(others[i]) != 0 or i != relation:
                        same_category = False
                        if i == relation:
                            same_category = True
                        if same_category:
                            target_size = 2 * output_disks_radii.shape[0] * output_disks_radii.shape[0]
                        else:
                            target_size = 2 * output_disks_radii.shape[0] * len(others[relation])

                        others_rel_torch = torch.stack(others[relation], 0)  # torch.Tensor: [N, 3]
                        weights_rel_torch = torch.stack(weights[relation], 0)
                        # print(others_rel_torch.shape, weights_rel_torch.shape)
                        # print(target_size)

                        test_pcf = compute_contribution(self.device, self.nSteps, d_test, others_rel_torch,
                                                        weights_rel_torch, cur_pcf_model, same_category,
                                                        target_size, diskfact)
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
                        test_pcf.weights = get_weight(d_test, cur_pcf_model, diskfact)

                    contributions[relation].pcf = test_pcf.pcf.clone()
                    contributions[relation].weights = test_pcf.weights.clone()
                    contributions[relation].contribution = test_pcf.contribution.clone()

                if rejected:
                    fails += 1
                else:
                    print('===> Before Gird Search, n_accepted: {:}/{:}'.format(n_accepted + 1,
                                                                                output_disks_radii.shape[0]))

                    others[i].append(d_test)
                    fails = 0
                    # if i == 1:
                    #     print('n_accepted:', n_accepted, d_test)
                    for relation in self.relations[i]:
                        current = current_pcf[relation]
                        contrib = contributions[relation]
                        if relation == i:
                            weights[relation].append(contrib.weights)
                        current_pcf[relation] += contrib.contribution.clone()
                        # if i == 1:
                        #     print(current_pcf[relation])
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
                                # cell_test = [gx, gy, output_disks_radii[n_accepted]]
                                # if min_x > gx > max_x or min_y > gy > max_y:
                                #     continue
                                # cell_test = torch.from_numpy(np.array(cell_test)).float().to(self.device)

                                cell_test = output_disks_radii[n_accepted].clone()
                                cell_test[:, 0] += gx
                                cell_test[:, 1] += gy

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
            Refinement step:
            """
            # TODO: refinement
            # category_i = i
            # current_output = torch.stack(others[category_i], 0)
            # model_refine = Refiner(self.device, self.opt, current_output).to(self.device)
            # optimizer = torch.optim.Adam(model_refine.parameters(), lr=1e-4)
            # n_iter = self.opt.n_iter
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

    def forward(self):
        """
        TODO: not implemented
        """
        pass
