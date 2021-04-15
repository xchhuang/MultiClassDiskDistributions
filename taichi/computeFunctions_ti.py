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
    def __init__(self, disks, disks_parent, same_category, nbbins=50, sigma=0.25, npoints=100, n_rmax=5, domainLength=1):
        super(PCF_ti, self).__init__()

        self.disks = disks
        self.disks_parent = disks_parent
        self.disks_ti = ti.Vector.field(3, dtype=ti.f32, shape=len(disks))
        self.disks_parent_ti = ti.Vector.field(3, dtype=ti.f32, shape=len(disks_parent))

        self.same_category = same_category
        self.pcf_mean = ti.Vector.field(2, dtype=ti.f32, shape=nbbins)
        self.pcf_min = ti.field(dtype=ti.f32, shape=nbbins)
        self.pcf_max = ti.field(dtype=ti.f32, shape=nbbins)
        self.domainLength = domainLength

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

        # self.rs = torch.from_numpy(np.array(rs)).float().to(device)
        # self.area = torch.from_numpy(np.array(area)).float().to(device)
        # self.sigma = torch.from_numpy(np.array([sigma])).float().to(device)
        # print('self.sigma:', self.sigma)
        self.sigma = sigma
        self.gf = 1.0 / (np.sqrt(math.pi) * self.sigma)

    def initialize(self):
        print('initialze')
        self.disks_ti.from_numpy(np.array(self.disks))
        self.disks_parent_ti.from_numpy(np.array(self.disks_parent))
        self.pcf_min.from_numpy(np.ones(self.nbbins) * np.inf)
        self.pcf_max.from_numpy(np.ones(self.nbbins) * -np.inf)

    def gaussianKernel(self, x):
        # return self.gf * torch.exp(-(x * x) / (self.sigma * self.sigma))
        # print(x)
        return self.gf * torch.exp(-((x * x) / (self.sigma * self.sigma)))

    def perimeter_weight(self, x_, y_, diskfact=1):

        full_angle = torch.ones(self.nbbins).float() * 2 * math.pi
        full_angle = full_angle.to(self.device)
        weights = torch.zeros_like(full_angle)
        rs = self.rs * diskfact
        # print(diskfact)
        x = x_ * diskfact
        y = y_ * diskfact
        # rs *= diskfact

        dx = x
        dy = y
        idx = rs > dx
        if rs.size(0) > 0:
            alpha = torch.acos(dx / rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        dx = 1 - x
        dy = y
        idx = rs > dx
        if rs[idx].size(0) > 0:
            alpha = torch.acos(dx / rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        dx = y
        dy = x
        idx = rs > dx
        if rs[idx].size(0) > 0:
            alpha = torch.acos(dx / rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx));

        dx = 1 - y
        dy = x
        idx = rs > dx
        if rs[idx].size(0) > 0:
            alpha = torch.acos(dx / rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        perimeter = torch.clamp(full_angle / (2 * math.pi), 0, 1)
        idx = perimeter > 0  # none zero
        weights[idx] = 1 / perimeter[idx]
        # weights = 1 / perimeter
        return weights

    @ti.kernel
    def forward_target(self):
        Na = self.disks_ti.shape[0]
        Nb = self.disks_parent_ti.shape[0]
        for i in range(Na):
            pi = self.disks_ti[i]
            for j in range(Nb):
                pj = self.disks_parent_ti[j]
                # print(pi, pj)
            # print(pi)
        #     print(pi.shape)
        #     if not same_category:
        #         pj = disks_b
        #         pi = pi.repeat(pj.size(0), 1)
        #     else:
        #         pj = torch.cat([disks_a[0:i], disks_a[i + 1:]])  # ignore the i_th disk itself
        #         pi = pi.repeat(pj.size(0), 1)
        #
        #     if use_fnorm:
        #         d = utils.diskDistance(pi, pj, self.rmax)
        #
        #         val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins))
        #         # print(self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins))
        #         # print((self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins)).shape)
        #         # print(val)
        #     else:
        #         # dis = utils.euclidean(pts_1, pts_2)
        #         # diff = (self.rs.view(1, -1) - dis.view(-1, 1).repeat(1, self.nbbins)) / self.rmax
        #         # val = self.gaussianKernel(diff)
        #         pass
        #
        #     pts_w = pi[0:1].view(1, -1)  # same
        #     weights = self.perimeter_weight(pts_w[:, 0], pts_w[:, 1], 1 / domainLength)
        #     density = torch.sum(val, 0)
        #     # print(density)
        #     density = density * weights / self.area  # consistent with cpp results until now
        #     # print(weights)
        #     density = density / disks_b.size(0)
        #     # print(density)
        #     pcf[:, 1] = pcf[:, 1] + density
        #     pcf_lower = torch.min(pcf_lower, density)
        #     pcf_upper = torch.max(pcf_upper, density)
        #
        # pcf[:, 1] = pcf[:, 1] / disks_a.size(0)
        # # print(pcf[:, 1])
        # pcf[:, 0] = self.rs / self.rmax






































































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
