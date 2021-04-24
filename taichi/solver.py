import taichi as ti
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import math
import utils
from time import time

ti.init(ti.cpu)


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

        graph, topological_order = utils.topologicalSort(len(self.categories.keys()), self.relations)
        print('topological_order:', topological_order)
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
        self.pcf_tmp_weights = ti.field(dtype=ti.f32, shape=(total_num_disks, self.nSteps))

        """
        Initialize taichi fields, after defining them all
        """
        prev_num_elem = 0
        prev_num_relation = 0
        for i in range(len(topological_order)):
            self.topological_order_ti[i] = topological_order[i]
            self.target_id2ElemNum[topological_order[i]] = len(categories[topological_order[i]])
            self.target_id2RelNum[topological_order[i]] = len(relations[topological_order[i]])
            self.target_id2PrevElemNum[topological_order[i]] = prev_num_elem
            self.target_id2PrevRelNum[topological_order[i]] = prev_num_relation
            prev_num_elem += len(categories[topological_order[i]])
            prev_num_relation += len(relations[topological_order[i]])
            rmax = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * len(categories[topological_order[i]])))
            self.target_rmax[topological_order[i]] = rmax
            step_size = self.n_rmax / self.nSteps
            for k in range(self.nSteps):
                radii = (i + 1) * step_size * rmax
                self.target_radii[topological_order[i], k] = radii
                inner = max(0, radii - 0.5 * rmax)
                outer = radii + 0.5 * rmax
                self.target_area[topological_order[i], k] = math.pi * (outer * outer - inner * inner)

        print('topological_order:', self.topological_order)
        print('id2num, prevNum:', self.target_id2ElemNum, self.target_id2RelNum, self.target_id2PrevElemNum,
              self.target_id2PrevRelNum)
        print('total_num_relations:', total_num_relations, self.relations)

        for i in range(len(topological_order)):
            num_relation = len(self.relations[topological_order[i]])
            for j in range(num_relation):
                for k in range(self.nSteps):
                    ind = (self.target_id2PrevRelNum[i] * self.nSteps) + (j * self.nSteps + k)

        start = time()
        for i in range(len(topological_order)):
            num_elem = self.target_id2ElemNum[self.topological_order[i]]
            for j in range(num_elem):
                for k in range(self.total_samples_per_element):
                    for l in range(3):
                        ind = self.target_id2PrevElemNum[i] * self.total_samples_per_element * 3 + (
                                j * self.total_samples_per_element * 3) + (k * 3 + l)
                        # print(ind)
                        self.target[ind] = self.categories[topological_order[i]][j][k][l]

        end = time()
        print('Time for initializing target elements: {:.4f}s'.format(end - start))
        utils.plot_disks_ti(self.topological_order_ti, self.target, self.target_id2ElemNum, self.target_id2PrevElemNum,
                            self.total_samples_per_element, self.opt.output_folder+'/target_element_ti')


        start = time()
        self.compute_target_pcf()
        end = time()
        print('Time for initializing target PCF: {:.4f}s'.format(end - start))

        return
        self.output = defaultdict(list)
        self.target_pcf_mean = defaultdict(dict)  # [id][id_parent][nSteps]
        self.target_pcf_min = defaultdict(dict)
        self.target_pcf_max = defaultdict(dict)
        self.target_pcf_radii = defaultdict(dict)
        self.target_pcf_rmax = defaultdict(dict)
        self.target_pcf_area = defaultdict(dict)
        self.target_pcf_count = defaultdict(dict)

        self.output_disks_radii = defaultdict(list)  # [id][num_out_disks_of_idclass], sort radii for synthesis
        self.current_pcf = defaultdict(list)  # [id_parent][nSteps]
        self.contributions = defaultdict(Contribution)  # [id_parent][Contribution]
        self.weights = defaultdict(list)  # [relation][num_output_disk, nSteps]
        self.count_disks = defaultdict(int)  # [id], return number of disks already added
        self.d_test_ti = ti.field(dtype=ti.f32, shape=3)  # [samplers_per_element, 3]
        self.test_pcf = Contribution(self.nSteps)  # [Contribution]
        self.tmp_weight = ti.field(dtype=ti.f32, shape=self.nSteps)  # tmp

        self.domainLength = opt.domainLength
        self.n_factor = self.domainLength * self.domainLength
        self.diskfact = 1
        # self.cur_id = 0
        # self.cur_parent_id = 0
        self.n_repeat = math.ceil(self.n_factor)

        for i in categories.keys():
            x = np.array(categories[i])
            if opt.samples_per_element:
                x = np.expand_dims(x, 1)

            # self.target[i] = ti.field(dtype=ti.f32, shape=x.shape)
            self.output[i] = ti.field(dtype=ti.f32, shape=(x.shape[0] * self.n_repeat, x.shape[1], x.shape[2]))
            self.output_disks_radii[i] = ti.field(dtype=ti.f32, shape=len(self.categories[i]) * self.n_repeat)
            self.current_pcf[i] = ti.field(dtype=ti.f32, shape=self.nSteps)
            self.contributions[i] = Contribution(self.nSteps)
            # self.weights[i] = ti.Vector.field(max_num_disks, dtype=ti.f32, shape=self.nSteps)
            for j in self.relations[i]:
                Nk = len(self.categories[i])
                same_category = False
                self.target_pcf_mean[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)
                self.target_pcf_min[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)
                self.target_pcf_max[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)
                self.target_pcf_count[i][j] = Nk
                self.target_pcf_rmax[i][j] = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * Nk))
                self.target_pcf_radii[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)
                self.target_pcf_area[i][j] = ti.field(dtype=ti.f32, shape=self.nSteps)

        for i in categories.keys():
            x = np.array(categories[i])
            if opt.samples_per_element:
                x = np.expand_dims(x, 1)
            print('x:', x.shape)  # [N, spe, 3]
            # self.target[i].from_numpy(x)
            self.output[i].from_numpy(np.zeros((x.shape[0] * self.n_repeat, x.shape[1], x.shape[2])))
            radii = x[:, 0, -1].repeat(self.n_repeat) / self.domainLength
            radii = np.sort(radii)[::-1]
            self.output_disks_radii[i].from_numpy(radii)
            self.current_pcf[i].from_numpy(np.zeros(self.nSteps))
            self.contributions[i].initialize()
            # self.weights[i] = ti.Vector.field(max_num_disks, dtype=ti.f32, shape=self.nSteps)
            for j in self.relations[i]:
                Nk = len(self.categories[i])
                same_category = False
                self.target_pcf_mean[i][j].from_numpy(np.zeros(self.nSteps))
                self.target_pcf_min[i][j].from_numpy(np.ones(self.nSteps) * np.inf)
                self.target_pcf_max[i][j].from_numpy(np.ones(self.nSteps) * -np.inf)
                for k in range(self.nSteps):
                    radii = (i + 1) * self.step_size * self.target_pcf_rmax[i][j]
                    self.target_pcf_radii[i][j][k] = radii
                    inner = max(0, radii - 0.5 * self.target_pcf_rmax[i][j])
                    outer = radii + 0.5 * self.target_pcf_rmax[i][j]
                    self.target_pcf_area[i][j][k] = math.pi * (outer * outer - inner * inner)

        start = time()
        self.compute_target_pcf()
        end = time()
        print('Time for computeTarget {:.4f}s.'.format(end - start))

    @ti.kernel
    def initialize_taichi_fields(self):
        pass

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

    @ti.kernel
    def compute_target_pcf_kernel(self, id_index: ti.i32, parent_id_index: ti.i32, same_category: ti.i32):
        # print('compute_target_pcf_kernel')
        num_elem_of_id = self.target_id2ElemNum[self.topological_order_ti[id_index]]
        num_elem_of_parent_id = self.target_id2ElemNum[self.topological_order_ti[parent_id_index]]
        # print(num_elem_of_id, num_elem_of_parent_id)
        rmax = self.target_rmax[self.topological_order_ti[id_index]]
        for i in ti.ndrange((0, num_elem_of_id)):
            for k in range(self.nSteps):
                self.weights_ti[i, k] = self.perimeter_weight_ti(pi[0], pi[1], self.rs_ti[k])

        for i, j in ti.ndrange((0, num_elem_of_id), (0, num_elem_of_parent_id)):
            skip = False
            if same_category and i == j:
                skip = True
            if not skip:
                for k1 in range(self.total_samples_per_element):
                    ind_i = self.target_id2PrevElemNum[id_index] * self.total_samples_per_element * 3 + (
                            i * self.total_samples_per_element * 3) + k1 * 3  # + l
                    p_i = ti.Vector([self.target[ind_i + 0], self.target[ind_i + 1], self.target[ind_i + 2]])

                    for k2 in range(self.total_samples_per_element):
                        ind_j = self.target_id2PrevElemNum[parent_id_index] * self.total_samples_per_element * 3 + (
                                j * self.total_samples_per_element * 3) + k2 * 3  # + l
                        p_j = ti.Vector([self.target[ind_j + 0], self.target[ind_j + 1], self.target[ind_j + 2]])
                        dist = self.diskDistance(p_i, p_j, rmax)
                        # print('dist:', dist)
            # print(i, j)
        #     for k in range(self.total_samples_per_element):
        #         for l in range(3):
        #             ind = prev_num_elem * self.total_samples_per_element * 3 + (
        #                     j * self.total_samples_per_element * 3) + (k * 3 + l)
        #             # print(ind)
        #             self.target[ind] = self.categories[topological_order[i]][j][k][l]
        #             # print(self.categories[topological_order[i]][j][k][l])
        # prev_num_elem += num_elem

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
                # cur_pcf_mean = self.target_pcf_model[i][j].pcf_mean.to_numpy()
                # # cur_pcf_mean2 = self.output_pcf_model[i][j].pcf_mean.to_numpy()
                #
                # plt.plot(cur_pcf_mean[:, 0], cur_pcf_mean[:, 1])
                # # print(cur_pcf_mean[:, 1])
                # plt.savefig(
                #     self.opt.output_folder + '/{:}_pcf_{:}_{:}'.format(self.opt.scene_name, i, j))
                # plt.clf()
                # return
