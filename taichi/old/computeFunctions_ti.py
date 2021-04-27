import numpy as np
import matplotlib.pyplot as plt
from time import time
import torch
import math
from tqdm import tqdm
import os
import glob
import utils
from utils import Contribution
import taichi as ti


def compute_contribution(device, nbbins, pi, others, other_weights, cur_pcf_model, same_category, target_size,
                         diskfactor):
    out = Contribution(device, nbbins)
    out.weights = cur_pcf_model.perimeter_weight(pi[0], pi[1], diskfactor)
    if len(others) == 0:
        return out
    pj = others
    pi = pi.unsqueeze(0).repeat(pj.size(0), 1)
    # print('pipj:', pi.shape, pj.shape)
    rmax = cur_pcf_model.rmax

    nbbins = cur_pcf_model.nbbins
    d = utils.diskDistance(pi, pj, rmax)
    rs = cur_pcf_model.rs
    area = cur_pcf_model.area
    # print(rs.shape, d.shape)
    val = cur_pcf_model.gaussianKernel(rs.view(1, -1) / rmax - d.view(-1, 1).repeat(1, nbbins))
    out.pcf = torch.sum(val, 0)
    # print(out.pcf)
    # print(res.shape, other_weights.shape)
    out.contribution = torch.sum(val * other_weights, 0)
    # print(out.contribution)
    out.pcf = out.pcf * out.weights / area
    out.contribution = out.pcf + out.contribution / area
    # print(len(others))
    out.pcf /= len(others)
    # print(out.contribution.shape, target_size)
    out.contribution /= target_size
    # print(out.pcf)
    return out


def filter_nan(x):
    '''
    :param x: torch.Tensor: (nbbins,)
    :return: torch.Tensor without nan(changed to -inf)
    :comment: use -inf so nan will not influence the mamximum value
    '''
    y = torch.where(torch.isnan(x), torch.full_like(x, -np.inf), x)
    # y = x
    # w_nan = torch.isnan(x)
    # print(torch.where(w_nan)[0].size(0), x)
    #
    # if torch.where(w_nan)[0].size(0) > 0:
    #     y = x[~torch.isnan(x)]
    return y


def compute_error(contribution, current_pcf, target_pcf):
    error_mean = -np.inf
    error_min = -np.inf
    error_max = -np.inf
    target_mean = target_pcf[:, 1]
    target_min = target_pcf[:, 2]
    target_max = target_pcf[:, 3]
    # print(target_pcf)

    x = (current_pcf + contribution.contribution - target_mean) / target_mean
    error_mean = max(filter_nan(x).max().item(), error_mean)
    # print(error_mean)
    x = (contribution.pcf - target_max) / target_max
    error_max = max(filter_nan(x).max().item(), error_max)

    x = (target_min - contribution.pcf) / target_min
    error_min = max(filter_nan(x).max().item(), error_min)
    # print('w/:', x)
    # print('wo/', filter_nan(x))
    ce = error_mean + max(error_max, error_min)
    # print(error_mean, error_min, error_max)
    return ce


@ti.data_oriented
class PCF_ti:
    def __init__(self, disks, disks_parent, same_category, nbbins=50, sigma=0.25, npoints=100, n_rmax=5,
                 domainLength=1):
        super(PCF_ti, self).__init__()

        n_factor = domainLength * domainLength
        npoints = int(npoints * domainLength)
        self.npoints = npoints
        self.n_repeat = math.ceil(n_factor)

        # self.count_disks = ti.field(dtype=ti.i32, shape=())    # TODO: counter for current number of disks, for output disks only instead of targets

        # print('n_repeat:', n_repeat)
        self.disks = disks
        self.disks_parent = disks_parent

        self.disks_ti = ti.Vector.field(3, dtype=ti.f32, shape=len(disks)*self.n_repeat)
        self.disks_parent_ti = ti.Vector.field(3, dtype=ti.f32, shape=len(disks_parent)*self.n_repeat)

        self.same_category = same_category
        self.pcf_mean = ti.Vector.field(2, dtype=ti.f32, shape=nbbins)
        self.pcf_min = ti.field(dtype=ti.f32, shape=nbbins)
        self.pcf_max = ti.field(dtype=ti.f32, shape=nbbins)
        self.rs_ti = ti.field(dtype=ti.f32, shape=nbbins)
        self.area_ti = ti.field(dtype=ti.f32, shape=nbbins)
        # self.weights_ti = ti.field(dtype=ti.f32, shape=nbbins)
        self.weights_ti = ti.field(dtype=ti.f32, shape=(npoints, nbbins))  # sacrifices space for parallel computation of PCF

        self.density = ti.field(dtype=ti.f32, shape=nbbins)

        self.domainLength = domainLength
        self.count_a = 0
        self.count_b = 0

        d = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * npoints))
        self.rmax = d
        self.nbbins = nbbins
        rs = []
        area = []
        stepsize = n_rmax / nbbins
        for i in range(nbbins):
            radii = (i + 1) * stepsize * self.rmax
            rs.append(radii)
            inner = max(0, radii - 0.5 * self.rmax)
            outer = radii + 0.5 * self.rmax
            area.append(math.pi * (outer * outer - inner * inner))

        self.rmin = rs[0]
        ra = rs[0]
        rb = rs[-1]
        # print('PCF Parameters:', self.rmin, self.rmax, npoints)

        self.sigma = sigma
        self.rs = rs
        self.area = area
        self.gf = 1.0 / (np.sqrt(math.pi) * self.sigma)

    def initialize(self):
        # print('initialze')
        # self.count_disks[None] = 0
        self.disks_ti.from_numpy(np.array(self.disks).repeat(self.n_repeat, 0))
        self.disks_parent_ti.from_numpy(np.array(self.disks_parent).repeat(self.n_repeat, 0))
        self.pcf_min.from_numpy(np.ones(self.nbbins) * np.inf)
        self.pcf_max.from_numpy(np.ones(self.nbbins) * -np.inf)
        # self.weights_ti.from_numpy(np.zeros(self.nbbins))
        self.weights_ti.from_numpy(np.zeros((self.npoints, self.nbbins)))
        self.area_ti.from_numpy(np.array(self.area))
        self.rs_ti.from_numpy(np.array(self.rs))

    @ti.func
    def gaussianKernel(self, x):
        return self.gf * ti.exp(-((x * x) / (self.sigma * self.sigma)))

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

    @ti.func
    def euclideanDistance(self, pi, pj):
        diff = pi - pj
        dist = ti.sqrt(diff[0] * diff[0] + diff[1] * diff[1])
        return dist

    @staticmethod
    @ti.func
    def clamp(x, lower, upper):
        return max(lower, min(upper, x))

    @staticmethod
    @ti.func
    def isnan(x):
        return not (x < 0 or 0 < x or x == 0)

    @ti.func
    def diskDistance(self, pi, pj):
        d = self.euclideanDistance(pi, pj) / self.rmax
        r1 = max(pi[2], pj[2]) / self.rmax
        r2 = min(pi[2], pj[2]) / self.rmax
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
    def forward(self):
        Na = self.disks_ti.shape[0]
        Nb = self.disks_parent_ti.shape[0]

        # for _ in range(1):      # TODO: force serialized, seems no longer needed
        for i in range(Na):
            pi = self.disks_ti[i]
            for k in range(self.nbbins):
                self.weights_ti[i, k] = self.perimeter_weight_ti(pi[0], pi[1], self.rs_ti[k])
            for j in range(Nb):
                skip = False
                if self.same_category and i == j:
                    skip = True
                if not skip:
                    pj = self.disks_parent_ti[j]
                    d = self.diskDistance(pi, pj)
                    for k in range(self.nbbins):
                        r = self.rs_ti[k] / self.rmax
                        val = self.gaussianKernel(r - d)
                        self.pcf_mean[k] += val * self.weights_ti[i, k] / self.area_ti[k] / Nb

            for k in range(self.nbbins):
                if self.pcf_min[k] > self.pcf_mean[k][1]:
                    self.pcf_min[k] = self.pcf_mean[k][1]
                if self.pcf_max[k] < self.pcf_mean[k][1]:
                    self.pcf_max[k] = self.pcf_mean[k][1]

        for k in range(self.nbbins):
            self.pcf_mean[k][0] = self.rs_ti[k] / self.rmax
            self.pcf_mean[k][1] /= Na

















































class PrettyPCF(torch.nn.Module):
    def __init__(self, device, nbbins=50, sigma=0.25, npoints=100, n_rmax=5):
        super(PrettyPCF, self).__init__()

        d = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * npoints))
        self.device = device
        self.rmax = d
        self.nbbins = nbbins
        rs = []
        area = []
        stepsize = n_rmax / nbbins
        for i in range(nbbins):
            radii = (i + 1) * stepsize * self.rmax
            rs.append(radii)
            inner = max(0, radii - 0.5 * self.rmax)
            outer = radii + 0.5 * self.rmax
            area.append(math.pi * (outer * outer - inner * inner))

        self.rmin = rs[0]
        ra = rs[0]
        rb = rs[-1]
        # print('PrettyPCF Parameters:', self.rmin, self.rmax, npoints)

        self.rs = torch.FloatTensor(np.array(rs)).to(device)
        self.area = torch.FloatTensor(np.array(area)).to(device)
        self.sigma = torch.FloatTensor([sigma]).to(device)
        self.gf = (1.0 / (np.sqrt(math.pi) * self.sigma))

    def gaussianKernel(self, x):
        return self.gf * torch.exp(-(x * x) / (self.sigma * self.sigma))

    def perimeter_weight(self, x, y):

        full_angle = torch.ones(self.nbbins) * 2 * math.pi
        full_angle = full_angle.to(self.device)
        weights = torch.zeros_like(full_angle)

        dx = x
        dy = y
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        dx = 1 - x
        dy = y
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        dx = y
        dy = x
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        dx = 1 - y
        dy = x
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        perimeter = torch.clamp(full_angle / (2 * math.pi), 0, 1)
        # idx = perimeter > 0  # none zero
        # weights[idx] = 1 / perimeter[idx]
        weights = 1 / perimeter
        return weights

    def forward(self, disks_a, disks_b, same_category, dimen=3, use_fnorm=True):
        # print(disks_a.shape, disks_b.shape, same_category)
        pcf = torch.zeros(self.nbbins, 2).to(self.device)
        # hist = torch.zeros(self.nbbins)
        pcf_lower = torch.ones(self.nbbins).to(self.device) * 1e4  # init with very large number
        pcf_upper = torch.zeros(self.nbbins).to(self.device)

        for i in range(disks_a.size(0)):
            pi = disks_a[i]
            if not same_category:
                pj = disks_b
                pi = pi.repeat(pj.size(0), 1)
            else:
                pj = torch.cat([disks_a[0:i], disks_a[i + 1:]])  # ignore the i_th disk itself
                pi = pi.repeat(pj.size(0), 1)

            d = torch.norm(pi[:, 0:2] - pj[:, 0:2], dim=1)  # only consider x, y
            # print('before:', d)
            # d = torch.sqrt(torch.sum((pi[:, 0:2] - pj[:, 0:2])**2, 1))
            # print('after:', d)
            # print(d)
            # val = self.gaussianKernel((self.rs - d) / self.rmax)
            val = self.gaussianKernel((self.rs.view(1, -1) - d.view(-1, 1).repeat(1, self.nbbins)) / self.rmax)
            density = torch.sum(val, 0)

            pts_w = pi[0:1].view(1, -1)  # same
            weights = self.perimeter_weight(pts_w[:, 0], pts_w[:, 1])
            weights = torch.clamp(weights, 0, 4)
            # weights = weights.unsqueeze(0)
            # print(val.shape, density.shape, weights.shape, pts_w.shape, weights.shape)

            pcf[:, 1] = pcf[:, 1] + density * weights / disks_a.size(0)
            # print(pcf[:, 1])
            # pcf_lower = torch.min(pcf_lower, density)
            # pcf_upper = torch.max(pcf_upper, density)
        pcf[:, 1] = pcf[:, 1] / (self.area * disks_b.size(0))
        # print(pcf[:, 1])
        pcf[:, 0] = self.rs / self.rmax
        return pcf


class PCF_utk(torch.nn.Module):
    def __init__(self, device, nbbins=50, sigma=0.01, npoints=100, n_rmax=5):
        super(PCF_utk, self).__init__()

        self.device = device
        d = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * npoints))
        self.rmax = d
        # self.rmax = 0.03
        self.nbbins = nbbins

        rs = []
        # area = []
        stepsize = n_rmax / nbbins
        for i in range(nbbins):
            radii = (i + 1) * stepsize * self.rmax
            rs.append(radii)
            inner = max(0, radii - 0.5 * self.rmax)
            outer = radii + 0.5 * self.rmax
            # area.append(math.pi * (outer * outer - inner * inner))

        self.rmin = rs[0]
        ra = rs[0]
        rb = rs[-1]
        # print('parameters:', self.rmin, self.rmax)

        rs = np.linspace(ra, rb, nbbins)
        self.rs = torch.FloatTensor(rs).to(device)
        self.sigma = torch.FloatTensor([sigma]).to(device)

        self.gf = (1.0 / (np.sqrt(2 * math.pi) * self.sigma))

    def gaussianKernel(self, x):
        return self.gf * torch.exp(-0.5 * (x * x) / (self.sigma * self.sigma))

    def forward(self, disks_a, disks_b, same_category):
        N = disks_a.size(0)
        pcf = torch.zeros(self.nbbins, 2).to(self.device)

        for i in range(N):
            pi = disks_a[i]
            if not same_category:
                pj = disks_b
            else:
                pj = torch.cat([disks_a[0:i], disks_a[i + 1:]], 0)
            pi = pi.repeat(pj.size(0), 1)
            diff = pi[:, 0:2] - pj[:, 0:2]  # [n,3]
            dis = torch.norm(diff, dim=-1)  # .view(-1,1)

            val = self.gaussianKernel(self.rs.view(1, -1) - dis.view(-1, 1).repeat(1, self.nbbins))
            # print (val.size())
            # cur_pcf[:,1] = cur_pcf[:,1] + torch.sum(val,0)
            pcf[:, 1] = pcf[:, 1] + torch.sum(val, 0)
            # print (torch.sum(val,1).size())

        cov = 1.0 - ((2.0 * self.rs) / math.pi) * 2.0 + ((self.rs * self.rs) / math.pi)
        factor = 2.0 * math.pi * self.rs * cov
        factor = 1.0 / factor
        pcf[:, 1] *= factor
        pcf[:, 1] = pcf[:, 1] / (disks_a.size(0) * disks_b.size(0))
        pcf[:, 0] = self.rs / self.rmax

        return pcf
