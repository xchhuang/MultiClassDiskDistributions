import taichi as ti
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import utils
from time import time

ti.init(ti.cpu)
# ti.init(ti.gpu)


@ti.data_oriented
class Contribution:
    def __init__(self, nbbins):
        self.nbbins = nbbins
        self.weights = ti.field(dtype=ti.f32, shape=nbbins)
        self.contribution = ti.field(dtype=ti.f32, shape=nbbins)
        self.pcf = ti.field(dtype=ti.f32, shape=nbbins)

    def initialize(self):
        self.weights.from_numpy(np.zeros(self.nbbins))
        self.contribution.from_numpy(np.zeros(self.nbbins))
        self.pcf.from_numpy(np.zeros(self.nbbins))


@ti.data_oriented
class Solver:
    def __init__(self, opt, categories, relations):
        super(Solver, self).__init__()
        self.opt = opt
        self.categories = categories
        self.relations = relations
        self.nSteps = opt.nSteps
        self.sigma = opt.sigma
        self.domainLength = opt.domainLength
        self.samples_per_element = opt.samples_per_element
        self.n_rmax = 5
        self.step_size = self.n_rmax / self.nSteps
        self.num_classes = len(categories.keys())
        self.total_samples_per_element = self.samples_per_element + 1
        self.domainLength = opt.domainLength
        self.n_factor = self.domainLength * self.domainLength
        self.disk_fact = 1 / self.domainLength
        self.n_repeat = math.ceil(self.n_factor)
        self.N_I = 100
        self.N_J = 100

        graph, topological_order = utils.topologicalSort(len(self.categories.keys()), self.relations)
        self.topological_order = topological_order

        max_num_disks = -np.inf
        total_num_relations = 0
        total_num_disks = 0
        for i in categories.keys():
            l = len(categories[i])
            total_num_disks += l
            total_num_relations += len(self.relations[i])
            if max_num_disks < l:
                max_num_disks = l

        """
        Define taichi fields, no initialization here
        """
        self.topological_order_ti = ti.field(dtype=ti.i32, shape=len(topological_order))
        self.relations_ti = ti.field(dtype=ti.i32, shape=total_num_relations)

        # samples_per_element + the center disk
        self.target = ti.field(dtype=ti.f32, shape=(total_num_disks * self.total_samples_per_element * 3))
        self.target_id2ElemNum = ti.field(dtype=ti.i32, shape=self.num_classes)
        self.target_id2PrevElemNum = ti.field(dtype=ti.i32, shape=self.num_classes)
        self.target_id2RelNum = ti.field(dtype=ti.i32, shape=self.num_classes)
        self.target_id2PrevRelNum = ti.field(dtype=ti.i32, shape=self.num_classes)
        self.target_pcf_mean = ti.field(ti.f32, shape=(total_num_relations * self.nSteps))
        self.target_pcf_min = ti.field(ti.f32, shape=(total_num_relations * self.nSteps))
        self.target_pcf_max = ti.field(ti.f32, shape=(total_num_relations * self.nSteps))
        self.target_rmax = ti.field(ti.f32, shape=self.num_classes)
        self.target_radii = ti.field(ti.f32, shape=(self.num_classes, self.nSteps))
        self.target_area = ti.field(ti.f32, shape=(self.num_classes, self.nSteps))
        self.pcf_tmp_weights = ti.field(dtype=ti.f32, shape=(total_num_disks * self.nSteps))
        self.pcf_density = ti.field(dtype=ti.f32, shape=(total_num_disks * self.nSteps))

        self.output = ti.field(dtype=ti.f32, shape=(total_num_disks * self.n_repeat * self.total_samples_per_element * 3))
        # self.output_id2ElemNum = ti.field(dtype=ti.i32, shape=self.num_classes)
        self.output_id2currentNum = ti.field(dtype=ti.i32, shape=self.num_classes)
        self.output_disks_radii = ti.field(dtype=ti.f32, shape=(total_num_disks * self.n_repeat * self.total_samples_per_element * 3))

        self.current_pcf = ti.field(dtype=ti.f32, shape=(self.num_classes, self.nSteps))
        self.weights = ti.field(dtype=ti.f32, shape=(self.num_classes, max_num_disks, self.nSteps))
        # Contribution struct: weights, contribution, pcf in order
        self.contributions = ti.field(dtype=ti.f32, shape=(self.num_classes, 3, self.nSteps))
        self.test_pcf = ti.field(dtype=ti.f32, shape=(3, self.nSteps))  # Contribution struct
        # for test_pcf(contribs[0,0]) and grid search
        self.contribs = ti.field(dtype=ti.f32, shape=(self.N_I, self.N_J, self.num_classes, 3, self.nSteps))
        self.errors = ti.field(dtype=ti.f32, shape=(self.N_I, self.N_J))

        self.minError_contrib = ti.field(dtype=ti.f32, shape=(self.num_classes, 3, self.nSteps))
        self.minError_i_ti = ti.field(dtype=ti.i32, shape=())
        self.minError_j_ti = ti.field(dtype=ti.i32, shape=())

        # self.current_contrib = ti.field(dtype=ti.f32, shape=(self.num_classes, 3, self.nSteps))
        # print(self.output.shape, self.output_id2currentNum)   # test num

        """
        Initialize taichi fields, after defining them all
        """

        prev_num_elem = 0
        prev_num_relation = 0
        for i in range(len(topological_order)):
            id = topological_order[i]
            self.topological_order_ti[i] = id
            self.target_id2ElemNum[id] = len(categories[id])
            self.target_id2RelNum[id] = len(relations[id])
            self.target_id2PrevElemNum[id] = prev_num_elem
            self.target_id2PrevRelNum[id] = prev_num_relation

            # self.output_id2ElemNum[id] = len(categories[id]) * self.n_repeat  # may be larger
            # self.output_id2currentNum[id] = 0

            prev_num_elem += len(categories[id])
            prev_num_relation += len(relations[id])
            rmax = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * len(categories[id])))
            self.target_rmax[id] = rmax
            step_size = self.n_rmax / self.nSteps
            for k in range(self.nSteps):
                radii = (k + 1) * step_size * rmax
                self.target_radii[topological_order[i], k] = radii
                inner = max(0, radii - 0.5 * rmax)
                outer = radii + 0.5 * rmax
                self.target_area[id, k] = math.pi * (outer * outer - inner * inner)

        # test relations_ti
        for i in range(len(topological_order)):
            id = topological_order[i]
            num_relation = len(relations[id])
            for j in range(num_relation):
                ind = self.target_id2PrevRelNum[id] + j
                self.relations_ti[ind] = relations[i][j]

        print('topological_order:', self.topological_order, self.topological_order_ti)
        print('id2num, prevNum:', self.target_id2ElemNum, self.target_id2RelNum, self.target_id2PrevElemNum,
              self.target_id2PrevRelNum)
        print('total_num_relations:', total_num_relations, self.relations, self.relations_ti)

        # test target
        start = time()
        for i in range(len(topological_order)):
            id = topological_order[i]
            num_elem = self.target_id2ElemNum[id]
            for j in range(num_elem):
                for k in range(self.total_samples_per_element):
                    for l in range(3):
                        ind = self.target_id2PrevElemNum[id] * self.total_samples_per_element * 3 + (j * self.total_samples_per_element * 3) + (k * 3 + l)
                        # print('ind:', ind)
                        self.target[ind] = self.categories[topological_order[i]][j][k][l]


        end = time()
        print('Time for initializing target elements: {:.4f}s'.format(end - start))
        utils.plot_disks_ti(self.topological_order_ti, self.target, self.target_id2ElemNum, self.target_id2PrevElemNum,
                            self.total_samples_per_element, 1, self.opt.output_folder + '/target_element_ti')

        # test for pcf_tmp_weights
        for i in range(len(self.topological_order)):
            id = self.topological_order[i]
            num_elem = self.target_id2ElemNum[id]
            for j in range(num_elem):
                for k in range(self.nSteps):
                    ind_w = self.target_id2PrevElemNum[id] * self.nSteps + j * self.nSteps + k
                    # print('ind_w:', ind_w)

        # test for target_pcf_*
        for i in range(len(self.topological_order)):
            id = self.topological_order[i]
            num_relation = self.target_id2RelNum[id]
            for j in range(num_relation):
                for k in range(self.nSteps):
                    ind_w = self.target_id2PrevRelNum[id] * self.nSteps + j * self.nSteps + k
                    # print('ind_w:', ind_w)
                    self.target_pcf_mean[ind_w] = 0.0
                    self.target_pcf_min[ind_w] = np.inf
                    self.target_pcf_max[ind_w] = -np.inf

        # for _ in range(2):
        start = time()
        self.compute_target_pcf()
        end = time()
        print('Time for initializing target PCF: {:.4f}s'.format(end - start))

        utils.plot_pcf(self.topological_order_ti, self.relations, self.target_pcf_mean, self.target_pcf_min, self.target_pcf_max, self.target_id2RelNum,
                       self.target_id2PrevRelNum,
                       self.nSteps, self.opt.output_folder + '/target_pcf')

        # test output_radii
        for i in range(len(topological_order)):
            id = self.topological_order[i]
            num_elem = self.target_id2ElemNum[self.topological_order[i]]
            radii = []
            for j in range(num_elem):
                k = 0
                l = 2
                ind = self.target_id2PrevElemNum[id] * self.total_samples_per_element * 3 + (
                        j * self.total_samples_per_element * 3) + (k * 3 + l)
                radii.append(self.target[ind])
                # self.target[ind] = self.categories[topological_order[i]][j][k][l]
            radii = np.array(radii)
            idx = np.argsort(radii)[::-1]
            # print(radii[idx])
            for j in range(len(idx) * self.n_repeat):
                # print(j, self.n_repeat, j // self.n_repeat)
                # print(j, idx[j])
                for k in range(self.total_samples_per_element):
                    for l in range(3):
                        ind_tar = self.target_id2PrevElemNum[id] * self.total_samples_per_element * 3 + (idx[j // self.n_repeat] * self.total_samples_per_element * 3) + (k * 3 + l)
                        ind_out = self.target_id2PrevElemNum[id] * self.n_repeat * self.total_samples_per_element * 3 + (j * self.total_samples_per_element * 3) + (k * 3 + l)
                        # print('ind_out:', ind_out)
                        # print('ind_tar:', ind_tar)

                        self.output_disks_radii[ind_out] = self.target[ind_tar] / self.domainLength

                # radii.append(self.target[ind])
            # print(self.output_disks_radii)
        utils.plot_disks_ti(self.topological_order_ti, self.output_disks_radii, self.target_id2ElemNum,
                            self.target_id2PrevElemNum,
                            self.total_samples_per_element, self.n_repeat,
                            self.opt.output_folder + '/test_output_element_ti')


        # for _ in range(2):
        self.initialization()

        utils.plot_disks_ti(self.topological_order_ti, self.output, self.target_id2ElemNum,
                            self.target_id2PrevElemNum,
                            self.total_samples_per_element, self.n_repeat,
                            self.opt.output_folder + '/output_element_ti')


    @ti.kernel
    def get_weights(self, id_index: ti.i32, parent_id_index: ti.i32):
        id = self.topological_order_ti[id_index]
        # parent_id = self.topological_order_ti[parent_id_index]
        parent_id = self.relations_ti[self.target_id2PrevRelNum[id] + parent_id_index]
        num_elem_of_parent_id = self.output_id2currentNum[parent_id]

        for i in range(num_elem_of_parent_id):
            ind = (self.target_id2PrevElemNum[id] * self.n_repeat) * self.total_samples_per_element * 3 + (
                    i * self.total_samples_per_element * 3) + (0 * 3 + 0)
            p_i = ti.Vector([self.output[ind + 0], self.output[ind + 1], self.output[ind + 2]])
            # get_weight
            for k in range(self.nSteps):
                val = self.perimeter_weight_ti(p_i[0], p_i[1], self.target_radii[id, k] * self.disk_fact)
                self.weights[parent_id, i, k] = val

    @ti.func
    def gaussianKernel(self, x):
        return (1.0 / (np.sqrt(math.pi) * self.sigma)) * ti.exp(-((x * x) / (self.sigma * self.sigma)))

    @ti.func
    def perimeter_weight_helper(self, dx, dy, r):
        angle = 0.0
        if dx < r:
            alpha = ti.acos(dx / r)
            angle = min(alpha, ti.atan2(dy, dx)) + min(alpha, ti.atan2((1 - dy), dx))
        return angle

    @ti.func
    def perimeter_weight_ti(self, x, y, r):
        full_angle = 2 * np.pi
        full_angle -= self.perimeter_weight_helper(x, y, r)
        full_angle -= self.perimeter_weight_helper(1 - x, y, r)
        full_angle -= self.perimeter_weight_helper(y, x, r)
        full_angle -= self.perimeter_weight_helper(1 - y, x, r)
        full_angle /= 2 * np.pi
        full_angle = self.clamp(full_angle, 0, 1)
        w = 0.0
        if full_angle > 0:
            w = 1.0 / full_angle
        return w

    @staticmethod
    @ti.func
    def isnan(x):
        return not (x < 0 or 0 < x or x == 0)

    @staticmethod
    @ti.func
    def clamp(x, lower, upper):
        return max(lower, min(upper, x))

    @ti.func
    def euclideanDistance(self, pi, pj):
        diff = pi - pj
        dist = ti.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
        return dist

    @ti.func
    def diskDistance(self, pi, pj, rmax):
        d = self.euclideanDistance(pi, pj) / rmax
        r1 = max(pi[2], pj[2]) / rmax
        r2 = min(pi[2], pj[2]) / rmax
        extent = max(d + r1 + r2, 2 * r1)
        overlap = self.clamp(r1 + r2 - d, 0, 2 * r2)
        f = extent - overlap + d + r1 - r2

        if d <= r1 - r2:
            f = f / (4 * r1 - 4 * r2)
        elif d <= r1 + r2:
            f = (f - 4 * r1 + 7 * r2) / (3 * r2)
        else:
            f = f - 4 * r1 - 2 * r2 + 3
        # if self.isnan(f):
        #     print(f)
        return f

    @ti.func
    def multiSphereDistance(self, cur_id, parent_id, ith_elem, jth_elem):
        d_outer = np.inf
        for k1 in range(self.total_samples_per_element):
            ind_i = self.target_id2PrevElemNum[cur_id] * self.total_samples_per_element * 3 + (ith_elem * self.total_samples_per_element * 3) + k1 * 3  # + l
            p_i = ti.Vector([self.target[ind_i + 0], self.target[ind_i + 1], self.target[ind_i + 2]])
            d_inner = np.inf
            for k2 in range(self.total_samples_per_element):
                ind_j = self.target_id2PrevElemNum[parent_id] * self.total_samples_per_element * 3 + (jth_elem * self.total_samples_per_element * 3) + k2 * 3  # + l
                p_j = ti.Vector([self.target[ind_j + 0], self.target[ind_j + 1], self.target[ind_j + 2]])
                dist = self.diskDistance(p_i, p_j, self.target_rmax[cur_id])
                if d_inner > dist:
                    d_inner = dist
            if d_outer > d_inner:
                d_outer = d_inner
        return d_outer

    @ti.kernel
    def compute_target_pcf_kernel(self, id_index: ti.i32, parent_id_index: ti.i32, same_category: ti.i32):
        # print('compute_target_pcf_kernel')
        id = self.topological_order_ti[id_index]
        # parent_id = self.topological_order_ti[parent_id_index]    # seems a stupid mistake
        parent_id = self.relations_ti[self.target_id2PrevRelNum[id] + parent_id_index]

        num_elem_of_id = self.target_id2ElemNum[id]
        num_elem_of_parent_id = self.target_id2ElemNum[parent_id]
        # print(num_elem_of_id, num_elem_of_parent_id)

        # for i, j in ti.ndrange((0, num_elem_of_id), (0, num_elem_of_parent_id)):
        for _ in range(1):
            for i in ti.ndrange(num_elem_of_id):

                for k in range(self.nSteps):
                    ind_i = self.target_id2PrevElemNum[id] * self.total_samples_per_element * 3 + (i * self.total_samples_per_element * 3) + 0 * 3 + 0  # + l
                    p_i = ti.Vector([self.target[ind_i + 0], self.target[ind_i + 1], self.target[ind_i + 2]])
                    ind_w = self.target_id2PrevElemNum[id] * self.nSteps + i * self.nSteps + k
                    # print('ind_w:', ind_w)
                    self.pcf_tmp_weights[ind_w] = self.perimeter_weight_ti(p_i[0], p_i[1], self.target_radii[id, k])    # target_radii[id] is correct
                    self.pcf_density[ind_w] = 0.0

                for j in ti.ndrange(num_elem_of_parent_id):
                    # skip = False
                    # if same_category and i == j:
                    #     skip = True
                    if (same_category and i != j) or (not same_category):
                        d_outer = self.multiSphereDistance(id, parent_id, i, j)

                        for k in range(self.nSteps):
                            r = self.target_radii[id, k] / self.target_rmax[id]
                            val = self.gaussianKernel(r - d_outer)
                            ind_w = self.target_id2PrevElemNum[id] * self.nSteps + i * self.nSteps + k
                            # print('ind_w:', ind_w)
                            # ind_pcf = self.target_id2PrevRelNum[id] * self.nSteps + parent_id_index * self.nSteps + k
                            # print('ind_pcf:', ind_pcf)
                            # self.target_pcf_mean[ind_pcf] += val * self.pcf_tmp_weights[ind_w] / self.target_area[id, k] / num_elem_of_parent_id
                            self.pcf_density[ind_w] += val * self.pcf_tmp_weights[ind_w] / self.target_area[id, k] / num_elem_of_parent_id

                for k in range(self.nSteps):
                    ind_pcf = self.target_id2PrevRelNum[id] * self.nSteps + parent_id_index * self.nSteps + k
                    ind_w = self.target_id2PrevElemNum[id] * self.nSteps + i * self.nSteps + k
                    self.target_pcf_mean[ind_pcf] += self.pcf_density[ind_w]
                    if self.target_pcf_min[ind_pcf] > self.pcf_density[ind_w]:
                        self.target_pcf_min[ind_pcf] = self.pcf_density[ind_w]
                    if self.target_pcf_max[ind_pcf] < self.pcf_density[ind_w]:
                        self.target_pcf_max[ind_pcf] = self.pcf_density[ind_w]

            for k in range(self.nSteps):
                ind_pcf = self.target_id2PrevRelNum[id] * self.nSteps + parent_id_index * self.nSteps + k
                # print('ind_pcf:', ind_pcf)
                self.target_pcf_mean[ind_pcf] /= num_elem_of_id

    def compute_target_pcf(self):
        for i in range(len(self.topological_order)):
            cur_id = self.topological_order[i]
            for j in range(len(self.relations[cur_id])):
                cur_parent_id = self.relations[cur_id][j]
                # print(cur_id, cur_parent_id)
                same_category = False
                if cur_id == cur_parent_id:
                    same_category = True
                self.compute_target_pcf_kernel(i, j, same_category)

    @ti.func
    def compute_contribution(self, grid_id_x, grid_id_y, id_index, parent_id_index, rx, ry, n_accepted, target_size):
        id = self.topological_order_ti[id_index]
        # parent_id = self.topological_order_ti[parent_id_index]
        parent_id = self.relations_ti[self.target_id2PrevRelNum[id] + parent_id_index]

        ind_cur = self.target_id2PrevElemNum[id] * self.n_repeat * self.total_samples_per_element * 3 + (n_accepted * self.total_samples_per_element * 3) + (0 * 3 + 0)  # k = 0, l = 0
        p_i = ti.Vector([rx, ry, self.output_disks_radii[ind_cur + 2]])  # random position (rx, ry)
        for k in range(self.nSteps):
            # self.test_pcf[0, k] = self.perimeter_weight_ti(p_i[0], p_i[1], self.target_radii[id, k] * self.disk_fact)
            self.contribs[grid_id_x, grid_id_y, parent_id, 0, k] = self.perimeter_weight_ti(p_i[0], p_i[1], self.target_radii[id, k] * self.disk_fact)
            self.contribs[grid_id_x, grid_id_y, parent_id, 1, k] = 0.0
            self.contribs[grid_id_x, grid_id_y, parent_id, 2, k] = 0.0

        is_continue = True
        if self.output_id2currentNum[parent_id] == 0:
            is_continue = False
        if is_continue:
            for j in range(self.output_id2currentNum[parent_id]):
                d_outer = np.inf
                for k1 in range(self.total_samples_per_element):
                    ind_i = self.target_id2PrevElemNum[id] * self.n_repeat * self.total_samples_per_element * 3 + (n_accepted * self.total_samples_per_element * 3) + k1 * 3  # + l
                    p_i = ti.Vector([rx, ry, self.output_disks_radii[ind_i + 2]])
                    d_inner = np.inf
                    for k2 in range(self.total_samples_per_element):
                        ind_j = self.target_id2PrevElemNum[parent_id] * self.n_repeat * self.total_samples_per_element * 3 + (j * self.total_samples_per_element * 3) + k2 * 3  # + l
                        p_j = ti.Vector([self.output[ind_j + 0], self.output[ind_j + 1], self.output[ind_j + 2]])
                        dist = self.diskDistance(p_i, p_j, self.target_rmax[id])    # target_rmax[id] should be indexed by [id]
                        if d_inner > dist:
                            d_inner = dist
                    if d_outer > d_inner:
                        d_outer = d_inner
                # print('d_outer:', d_outer)
                for k in range(self.nSteps):
                    r = self.target_radii[id, k] / self.target_rmax[id]
                    val = self.gaussianKernel(r - d_outer)
                    # print(val)
                    self.contribs[grid_id_x, grid_id_y, parent_id, 2, k] += val
                    self.contribs[grid_id_x, grid_id_y, parent_id, 1, k] += val * self.weights[parent_id, j, k]


            for k in range(self.nSteps):
                self.contribs[grid_id_x, grid_id_y, parent_id, 2, k] *= self.contribs[grid_id_x, grid_id_y, parent_id, 0, k] / self.target_area[id, k]
                self.contribs[grid_id_x, grid_id_y, parent_id, 1, k] = self.contribs[grid_id_x, grid_id_y, parent_id, 2, k] + \
                                                                       self.contribs[grid_id_x, grid_id_y, parent_id, 1, k] / self.target_area[id, k]
                self.contribs[grid_id_x, grid_id_y, parent_id, 2, k] /= self.output_id2currentNum[parent_id]
                self.contribs[grid_id_x, grid_id_y, parent_id, 1, k] /= target_size

    @ti.func
    def compute_error(self, grid_id_x, grid_id_y, id_index, parent_id_index):
        id = self.topological_order_ti[id_index]
        # parent_id = self.topological_order_ti[parent_id_index]
        parent_id = self.relations_ti[self.target_id2PrevRelNum[id] + parent_id_index]
        error_mean = -np.inf
        error_min = -np.inf
        error_max = -np.inf
        ind_w = self.target_id2PrevRelNum[id] * self.nSteps + parent_id_index * self.nSteps + 0
        for k in range(self.nSteps):
            x = (self.current_pcf[parent_id, k] + self.contribs[grid_id_x, grid_id_y, parent_id, 1, k] - self.target_pcf_mean[ind_w + k]) / self.target_pcf_mean[ind_w + k]
            # print('x:', x)
            if not self.isnan(x):
                error_mean = max(x, error_mean)

        for k in range(self.nSteps):
            x = (self.contribs[grid_id_x, grid_id_y, parent_id, 2, k] - self.target_pcf_max[ind_w + k]) / self.target_pcf_max[ind_w + k]
            # print('x:', x)
            if not self.isnan(x):
                error_max = max(x, error_max)

        for k in range(self.nSteps):
            x = (self.target_pcf_min[ind_w + k] - self.contribs[grid_id_x, grid_id_y, parent_id, 2, k]) / self.target_pcf_min[ind_w + k]
            # print('x:', x)
            if not self.isnan(x):
                error_min = max(x, error_min)

        ce = error_mean + max(error_max, error_min)
        # print('ce:', ce)
        return ce

    @ti.kernel
    def random_search(self, id_index: ti.i32, e: ti.f32, rx: ti.f32, ry: ti.f32, n_accepted: ti.i32) -> ti.i32:
        for _ in range(1):  # force serialization for now
            rejected = False
            id = self.topological_order_ti[id_index]
            num_relation = self.target_id2RelNum[id]
            for j in range(num_relation):
                ind_parent_id = self.target_id2PrevRelNum[id] + j
                parent_id = self.relations_ti[ind_parent_id]
                # print('ids:', id, parent_id)

                if self.output_id2currentNum[id] != 0 or id != parent_id:
                    # print('if')
                    same_category = False
                    if id == parent_id:
                        same_category = True
                    target_size = 2 * (self.target_id2ElemNum[id] * self.n_repeat) * self.output_id2currentNum[parent_id]
                    if same_category:
                        target_size = 2 * (self.target_id2ElemNum[id] * self.n_repeat) * (self.target_id2ElemNum[id] * self.n_repeat)

                    # print(rx, ry, target_size)
                    self.compute_contribution(0, 0, id_index, j, rx, ry, n_accepted, target_size)

                    ce = self.compute_error(0, 0, id_index, j)

                    if e < ce:
                        rejected = True
                        break

                else:
                    # print('else')
                    ind = (self.target_id2PrevElemNum[id] * self.n_repeat) * self.total_samples_per_element * 3 + (n_accepted * self.total_samples_per_element * 3) + (0 * 3 + 0)
                    p_i = ti.Vector([rx, ry, self.output_disks_radii[ind + 2]])
                    for k in range(self.nSteps):
                        val = self.perimeter_weight_ti(p_i[0], p_i[1], self.target_radii[id, k] * self.disk_fact)
                        self.contribs[0, 0, parent_id, 0, k] = val
                        self.contribs[0, 0, parent_id, 1, k] = 0.0
                        self.contribs[0, 0, parent_id, 2, k] = 0.0

                # TODO: this might be unnecessary
                for k in range(self.nSteps):
                    self.contributions[parent_id, 0, k] = self.contribs[0, 0, parent_id, 0, k]
                    self.contributions[parent_id, 1, k] = self.contribs[0, 0, parent_id, 1, k]
                    self.contributions[parent_id, 2, k] = self.contribs[0, 0, parent_id, 2, k]

            return rejected

    @ti.kernel
    def grid_search(self, id_index: ti.i32, n_accepted: ti.i32):
        minError_val = np.inf
        # minError_i = 1
        # minError_j = 1
        for n_i, n_j in ti.ndrange((1, self.N_I), (1, self.N_J)):
            id = self.topological_order_ti[id_index]
            num_relation = self.target_id2RelNum[id_index]
            gx = (1 / self.N_I) * n_i
            gy = (1 / self.N_J) * n_j
            # currentError = 0.0
            for j in range(num_relation):
                ind_parent_id = self.target_id2PrevRelNum[id] + j
                parent_id = self.relations_ti[ind_parent_id]
                same_category = False
                if id == parent_id:
                    same_category = True
                target_size = (self.target_id2ElemNum[id] * self.n_repeat) * self.output_id2currentNum[id]
                if same_category:
                    target_size = (self.target_id2ElemNum[id] * self.n_repeat) * (self.target_id2ElemNum[id] * self.n_repeat)
                self.compute_contribution(n_i, n_j, id_index, j, gx, gy, n_accepted, target_size)
                ce = self.compute_error(n_i, n_j, id_index, j)
                self.errors[n_i, n_j] = max(ce, self.errors[n_i, n_j])

        for n_i, n_j in ti.ndrange((1, self.N_I), (1, self.N_J)):
            if minError_val > self.errors[n_i, n_j]:
                minError_val = self.errors[n_i, n_j]
                # minError_i = n_i
                # minError_j = n_j
                self.minError_i_ti[None] = n_i
                self.minError_j_ti[None] = n_j

    def initialization(self):
        e_delta = 1e-4
        e_0 = 0
        max_fails = 1000


        for i in range(self.num_classes):
            self.output_id2currentNum[i] = 0

        start_time = time()
        for i in range(len(self.topological_order)):
            # start_time = time()

            cur_id = self.topological_order[i]
            fails = 0

            self.current_pcf.from_numpy(np.zeros((self.num_classes, self.nSteps)))
            self.contributions.from_numpy(np.zeros((self.num_classes, 3, self.nSteps)))

            for j in range(len(self.relations[cur_id])):
                # parent_id = self.relations[cur_id][j]
                self.get_weights(i, j)

            n_accepted = 0
            output_size = self.target_id2ElemNum[cur_id] * self.n_repeat
            # print('output_size:', output_size)


            while n_accepted < output_size:

                e = e_0 + e_delta * fails

                rx = np.random.rand()
                ry = np.random.rand()
                if i == 0:
                    min_xy = 0.05
                    rx = min_xy + np.random.rand() * (1 - min_xy * 2)  # my version
                    ry = min_xy + np.random.rand() * (1 - min_xy * 2)

                rejected = self.random_search(i, e, rx, ry, n_accepted)
                # print('rejected:', rejected)

                if rejected:
                    fails += 1
                else:
                    fails = 0
                    self.output_id2currentNum[cur_id] += 1
                    Ne = self.target_id2PrevElemNum[cur_id] * self.n_repeat
                    # print('Ne:', Ne)
                    for k in range(self.total_samples_per_element):
                        # for l in range(3):
                        ind_out = Ne * self.total_samples_per_element * 3 + (n_accepted * self.total_samples_per_element * 3) + (k * 3 + 0)
                        self.output[ind_out + 0] = rx
                        self.output[ind_out + 1] = ry
                        self.output[ind_out + 2] = self.output_disks_radii[ind_out + 2]
                        # print('self.output_disks_radii[ind_out + 2]:', self.output_disks_radii[ind_out + 2])

                    for j in range(len(self.relations[cur_id])):
                        parent_id = self.relations[cur_id][j]
                        # print(cur_id, parent_id)
                        if cur_id == parent_id:
                            for k in range(self.nSteps):
                                self.weights[parent_id, n_accepted, k] = self.contributions[parent_id, 0, k]
                        for k in range(self.nSteps):
                            self.current_pcf[parent_id, k] += self.contributions[parent_id, 1, k]

                    n_accepted += 1

                if fails > max_fails:
                    print('===> Warning: exceeding max_fails...', n_accepted)
                    # print('n_accepted before grid search:', n_accepted)
                    # break

                    while n_accepted < output_size:
                        # print(n_accepted, cur_id)
                        self.minError_i_ti[None] = 1
                        self.minError_j_ti[None] = 1
                        self.errors.from_numpy(np.zeros((self.N_I, self.N_J)))
                        # print('before:', n_accepted, self.minError_i_ti[None], self.minError_j_ti[None])
                        self.grid_search(i, n_accepted)
                        # print('after:', n_accepted, self.minError_i_ti[None], self.minError_j_ti[None])
                        # print(self.minError_i_ti.to_numpy())
                        minError_i = self.minError_i_ti.to_numpy()
                        minError_j = self.minError_j_ti.to_numpy()

                        # print(minError_i, minError_j)
                        rx = (1 / self.N_I) * minError_i + (np.random.rand() - 1 / 2) / (self.N_I * 10)
                        ry = (1 / self.N_J) * minError_j + (np.random.rand() - 1 / 2) / (self.N_J * 10)

                        self.output_id2currentNum[cur_id] += 1
                        Ne = self.target_id2PrevElemNum[cur_id] * self.n_repeat
                        for k in range(self.total_samples_per_element):
                            ind_out = Ne * self.total_samples_per_element * 3 + (n_accepted * self.total_samples_per_element * 3) + (k * 3 + 0)
                            self.output[ind_out + 0] = rx
                            self.output[ind_out + 1] = ry
                            self.output[ind_out + 2] = self.output_disks_radii[ind_out + 2]

                        num_relation = self.target_id2RelNum[cur_id]
                        for j in range(num_relation):
                            ind_parent_id = self.target_id2PrevRelNum[cur_id] + j
                            parent_id = self.relations_ti[ind_parent_id]
                            if cur_id == parent_id:
                                for k in range(self.nSteps):
                                    self.weights[parent_id, n_accepted, k] = self.contribs[minError_i, minError_j, parent_id, 0, k]
                            for k in range(self.nSteps):
                                self.current_pcf[parent_id, k] += self.contribs[minError_i, minError_j, parent_id, 1, k]

                        n_accepted += 1

            # end_time = time()
            # print('Initialization time of id {:} is {:.4f}s'.format(cur_id, end_time - start_time))

        end_time = time()
        print('Initialization time is: {:.4f}s'.format(end_time - start_time))
