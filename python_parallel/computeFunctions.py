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


def compute_contribution(device, nbbins, pi, others, other_weights, cur_pcf_model, same_category, target_size, diskfactor):
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


class PCF(torch.nn.Module):
    def __init__(self, device, nbbins=50, sigma=0.25, npoints=100, n_rmax=5):
        super(PCF, self).__init__()

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
        # print('PCF Parameters:', self.rmin, self.rmax, npoints)

        self.rs = torch.from_numpy(np.array(rs)).float().to(device)
        self.area = torch.from_numpy(np.array(area)).float().to(device)
        # self.sigma = torch.from_numpy(np.array([sigma])).float().to(device)
        # print('self.sigma:', self.sigma)
        self.sigma = sigma
        self.gf = 1.0 / (np.sqrt(math.pi) * self.sigma)

    def gaussianKernel(self, x):
        # return self.gf * torch.exp(-(x * x) / (self.sigma * self.sigma))
        # print(x)
        return self.gf * torch.exp(-((x * x)/(self.sigma * self.sigma)))

    def perimeter_weight(self, x_, y_, diskfact = 1):

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

    # def compute_weights(self, gen_pts, relation):
    #     # print (len(gen_pts))
    #     gen_pts = torch.FloatTensor(np.array(gen_pts))
    #     pts = gen_pts[gen_pts[:, 3] == relation]
    #     other_weights = []
    #     for i in range(pts.size(0)):
    #         p = pts[i:i + 1]
    #         weights = 1 / self.perimeter_weight(p[:, 0], p[:, 1])
    #         # print(weights)
    #         other_weights.append(weights)
    #     other_weights = torch.stack(other_weights, dim=0)
    #     # print (other_weights[0], other_weights.size())
    #     return other_weights
    #
    # def compute_contribution(self, disk, gen_pts, other_weights, target_pts, id, relation):
    #
    #     other_weights = other_weights[relation]
    #     gen_pts = torch.FloatTensor(np.array(gen_pts))
    #     other = gen_pts[gen_pts[:, 3] == relation]
    #     cur = gen_pts[gen_pts[:, 3] == id]
    #
    #     denorm_1 = target_pts[target_pts[:, 3] == id].size(0)
    #     denorm_2 = target_pts[target_pts[:, 3] == relation].size(0)
    #     denom = denorm_1 * denorm_2
    #
    #     # print (gen_pts, other)
    #     disk = torch.FloatTensor(np.array(disk)).unsqueeze(0)
    #     # print (disk.size())
    #     disk = disk.repeat(other.size(0), 1)
    #
    #     # pcf = torch.zeros(self.nbbins)
    #     # weights = torch.zeros(self.nbbins)
    #     # contribution = torch.zeros(self.nbbins)
    #
    #     # for i in range(other.size(0)):
    #
    #     dis = utils.fnorm(disk[:, 0:3], other[:, 0:3], self.rmax)
    #     # print (self.rmax)
    #     # print ('disk:',disk[:,0:3])
    #     # print ('other:',other[:,0:3])
    #     # print (dis)
    #     # print (other_weights)
    #
    #     val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - dis.view(-1, 1).repeat(1, self.nbbins))
    #
    #     weights = 1 / self.perimeter_weight(disk[0, 0], disk[0, 1])
    #
    #     # print (val.size(), other_weights.size())
    #
    #     contribution = torch.sum(other_weights * val, 0)
    #     # torch.set_printoptions(precision=8)
    #     # print (contribution,denom)
    #
    #     hist = torch.sum(weights * val, 0)
    #     pcf = hist / self.area
    #
    #     contribution = pcf + contribution / self.area
    #
    #     pcf = pcf / other.size(0)
    #     # print (pcf)
    #     contribution = contribution / denom
    #     # print (contribution,denom)
    #     return contribution, pcf
    #
    # def compute_error(self, contribution, test_pcf, target_pcf, relation):
    #     E = 0
    #     eps = 1e-7
    #     error_mean = 0
    #     error_min = 0
    #     error_max = 0
    #
    #     # pcf0 = target_pcf[0]
    #     # pcf1 = target_pcf[1]
    #     # target_pcf_mean = torch.mean(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)
    #     # target_pcf_max = torch.max(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)[0]
    #     # target_pcf_min = torch.min(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)[0]
    #     target_pcf_mean = target_pcf[relation][:, 1]
    #     target_pcf_min = target_pcf[relation][:, 2]
    #     target_pcf_max = target_pcf[relation][:, 3]
    #
    #     # print (torch.max((test_pcf - target_pcf_max)/(target_pcf_max+eps)))
    #     error_mean = max(((contribution - target_pcf_mean) / (target_pcf_mean + eps)).max(), error_mean)
    #     # print ('error_mean: ', error_mean)
    #     error_max = max(((test_pcf - target_pcf_max) / (target_pcf_max + eps)).max(), error_max)
    #     # print ('error_max: ', error_max)
    #     error_min = max(((target_pcf_min - test_pcf) / (target_pcf_min + eps)).max(), error_min)
    #     # print (error_min)
    #     E = error_mean + max(error_max, error_min);
    #     # print (E)
    #     return E

    # def forward(self, pts, c, t, dimen=3, use_fnorm=True, mean_pcf=False):
    def forward(self, disks_a, disks_b, same_category, dimen=3, use_fnorm=True, domainLength=1):
        # print(disks_a.shape, disks_b.shape, same_category)
        pcf = torch.zeros(self.nbbins, 2).float().to(self.device)
        # hist = torch.zeros(self.nbbins)
        pcf_lower = torch.ones(self.nbbins).float().to(self.device) * np.inf  # init with very large number
        pcf_upper = torch.ones(self.nbbins).float().to(self.device) * -np.inf

        for i in range(disks_a.size(0)):
            pi = disks_a[i]
            if not same_category:
                pj = disks_b
                pi = pi.repeat(pj.size(0), 1)
            else:
                pj = torch.cat([disks_a[0:i], disks_a[i+1:]])   # ignore the i_th disk itself
                pi = pi.repeat(pj.size(0), 1)

            if use_fnorm:
                d = utils.diskDistance(pi, pj, self.rmax)
                # print(i, d)
                val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins))
                # print(self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins))
                # print((self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins)).shape)
                # print(val)
            else:
                # dis = utils.euclidean(pts_1, pts_2)
                # diff = (self.rs.view(1, -1) - dis.view(-1, 1).repeat(1, self.nbbins)) / self.rmax
                # val = self.gaussianKernel(diff)
                pass

            pts_w = pi[0:1].view(1, -1)  # same
            weights = self.perimeter_weight(pts_w[:, 0], pts_w[:, 1], 1/domainLength)
            density = torch.sum(val, 0)

            density = density * weights / self.area  # consistent with cpp results until now
            # print(weights)
            density = density / disks_b.size(0)
            # print(density)
            pcf[:, 1] = pcf[:, 1] + density
            pcf_lower = torch.min(pcf_lower, density)
            pcf_upper = torch.max(pcf_upper, density)

        pcf[:, 1] = pcf[:, 1] / disks_a.size(0)
        # print(pcf[:, 1])
        pcf[:, 0] = self.rs / self.rmax
        return pcf, pcf_lower, pcf_upper












































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
                pj = torch.cat([disks_a[0:i], disks_a[i+1:]])   # ignore the i_th disk itself
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
                pj = torch.cat([disks_a[0:i], disks_a[i+1:]], 0)
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


