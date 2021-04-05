import numpy as np
import matplotlib.pyplot as plt
from time import time
import torch
import math
from tqdm import tqdm
import os
import glob
import utils


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
        print('PCF Parameters:', self.rmin, self.rmax, npoints)

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
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx));

        dx = y
        dy = x
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx));

        dx = 1 - y
        dy = x
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        perimeter = torch.clamp(full_angle / (2 * math.pi), 0, 1)
        idx = perimeter > 1e-4  # none zero
        weights[idx] = 1 / perimeter[idx]
        return weights

    def compute_weights(self, gen_pts, relation):
        # print (len(gen_pts))
        gen_pts = torch.FloatTensor(np.array(gen_pts))
        pts = gen_pts[gen_pts[:, 3] == relation]
        other_weights = []
        for i in range(pts.size(0)):
            p = pts[i:i + 1]
            weights = 1 / self.perimeter_weight(p[:, 0], p[:, 1])
            # print(weights)
            other_weights.append(weights)
        other_weights = torch.stack(other_weights, dim=0)
        # print (other_weights[0], other_weights.size())
        return other_weights

    def compute_contribution(self, disk, gen_pts, other_weights, target_pts, id, relation):

        other_weights = other_weights[relation]
        gen_pts = torch.FloatTensor(np.array(gen_pts))
        other = gen_pts[gen_pts[:, 3] == relation]
        cur = gen_pts[gen_pts[:, 3] == id]

        denorm_1 = target_pts[target_pts[:, 3] == id].size(0)
        denorm_2 = target_pts[target_pts[:, 3] == relation].size(0)
        denom = denorm_1 * denorm_2

        # print (gen_pts, other)
        disk = torch.FloatTensor(np.array(disk)).unsqueeze(0)
        # print (disk.size())
        disk = disk.repeat(other.size(0), 1)

        # pcf = torch.zeros(self.nbbins)
        # weights = torch.zeros(self.nbbins)
        # contribution = torch.zeros(self.nbbins)

        # for i in range(other.size(0)):

        dis = utils.fnorm(disk[:, 0:3], other[:, 0:3], self.rmax)
        # print (self.rmax)
        # print ('disk:',disk[:,0:3])
        # print ('other:',other[:,0:3])
        # print (dis)
        # print (other_weights)

        val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - dis.view(-1, 1).repeat(1, self.nbbins))

        weights = 1 / self.perimeter_weight(disk[0, 0], disk[0, 1])

        # print (val.size(), other_weights.size())

        contribution = torch.sum(other_weights * val, 0)
        # torch.set_printoptions(precision=8)
        # print (contribution,denom)

        hist = torch.sum(weights * val, 0)
        pcf = hist / self.area

        contribution = pcf + contribution / self.area

        pcf = pcf / other.size(0)
        # print (pcf)
        contribution = contribution / denom
        # print (contribution,denom)
        return contribution, pcf

    def compute_error(self, contribution, test_pcf, target_pcf, relation):
        E = 0
        eps = 1e-7
        error_mean = 0
        error_min = 0
        error_max = 0

        # pcf0 = target_pcf[0]
        # pcf1 = target_pcf[1]
        # target_pcf_mean = torch.mean(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)
        # target_pcf_max = torch.max(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)[0]
        # target_pcf_min = torch.min(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)[0]
        target_pcf_mean = target_pcf[relation][:, 1]
        target_pcf_min = target_pcf[relation][:, 2]
        target_pcf_max = target_pcf[relation][:, 3]

        # print (torch.max((test_pcf - target_pcf_max)/(target_pcf_max+eps)))
        error_mean = max(((contribution - target_pcf_mean) / (target_pcf_mean + eps)).max(), error_mean)
        # print ('error_mean: ', error_mean)
        error_max = max(((test_pcf - target_pcf_max) / (target_pcf_max + eps)).max(), error_max)
        # print ('error_max: ', error_max)
        error_min = max(((target_pcf_min - test_pcf) / (target_pcf_min + eps)).max(), error_min)
        # print (error_min)
        E = error_mean + max(error_max, error_min);
        # print (E)
        return E

    # def forward(self, pts, c, t, dimen=3, use_fnorm=True, mean_pcf=False):
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

            if use_fnorm:
                d = utils.diskDistance(pi, pj, self.rmax)
                val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins))
                # print(val.shape, val[90:])
            else:
                # dis = utils.euclidean(pts_1, pts_2)
                # diff = (self.rs.view(1, -1) - dis.view(-1, 1).repeat(1, self.nbbins)) / self.rmax
                # val = self.gaussianKernel(diff)
                pass

            pts_w = pi[0:1].view(1, -1)  # same
            weights = self.perimeter_weight(pts_w[:, 0], pts_w[:, 1])
            # print(weights)
            density = torch.sum(val * weights / self.area, 0)   # consistent with cpp results until now
            density = density / disks_b.size(0)
            pcf[:, 1] = pcf[:, 1] + density
            pcf_lower = torch.min(pcf_lower, density)
            pcf_upper = torch.max(pcf_upper, density)

        pcf[:, 1] = pcf[:, 1] / disks_a.size(0)
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
        print('PCF Parameters:', self.rmin, self.rmax, npoints)

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
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx));

        dx = y
        dy = x
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx));

        dx = 1 - y
        dy = x
        idx = self.rs > dx
        if self.rs[idx].size(0) > 0:
            alpha = torch.acos(dx / self.rs[idx])
            full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))

        perimeter = torch.clamp(full_angle / (2 * math.pi), 0, 1)
        idx = perimeter > 1e-4  # none zero
        weights[idx] = 1 / perimeter[idx]
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

            if use_fnorm:
                d = utils.diskDistance(pi, pj, self.rmax)
                val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - d.view(-1, 1).repeat(1, self.nbbins))
                # print(val.shape, val[90:])
            else:
                # dis = utils.euclidean(pts_1, pts_2)
                # diff = (self.rs.view(1, -1) - dis.view(-1, 1).repeat(1, self.nbbins)) / self.rmax
                # val = self.gaussianKernel(diff)
                pass

            pts_w = pi[0:1].view(1, -1)  # same
            weights = self.perimeter_weight(pts_w[:, 0], pts_w[:, 1])
            # print(weights)
            density = torch.sum(val * weights / self.area, 0)   # consistent with cpp results until now
            density = density / disks_b.size(0)
            pcf[:, 1] = pcf[:, 1] + density
            pcf_lower = torch.min(pcf_lower, density)
            pcf_upper = torch.max(pcf_upper, density)

        pcf[:, 1] = pcf[:, 1] / disks_a.size(0)
        pcf[:, 0] = self.rs / self.rmax
        return pcf, pcf_lower, pcf_upper





















































































# class PCFLearner(torch.nn.Module):
#     def __init__(self, init_pts, ndim=2, npoints=700):
#         super(PCFLearner, self).__init__()
#
#         self.use_mlp = False
#         self.n_samples = init_pts.shape[0]
#         self.w = torch.nn.Parameter(torch.FloatTensor(init_pts[:, 0:2]))
#         # self.w = torch.nn.Parameter(torch.zeros(init_pts.shape[0],2))
#         if self.use_mlp:
#             self.w.requires_grad = False  # True
#         else:
#             self.w.requires_grad = True  # True
#
#         self.w2 = torch.nn.Parameter(torch.FloatTensor(init_pts[:, 2:4]))
#         self.w2.requires_grad = False
#
#         # self.mask1 = torch.nn.Parameter(torch.zeros(10,2))
#         # self.mask1.requires_grad = False
#         # self.mask2 = torch.nn.Parameter(torch.ones(90,2))
#         # self.mask2.requires_grad = False
#         h = 128
#         self.mlp = torch.nn.Sequential(
#             torch.nn.Linear(180, h),
#             # torch.nn.BatchNorm1d(180),
#             torch.nn.LeakyReLU(0.2, True),
#             torch.nn.Linear(h, h),
#             torch.nn.LeakyReLU(0.2, True),
#             torch.nn.Linear(h, h),
#             torch.nn.LeakyReLU(0.2, True),
#             torch.nn.Linear(h, h),
#             torch.nn.LeakyReLU(0.2, True),
#             torch.nn.Linear(h, h),
#             torch.nn.LeakyReLU(0.2, True),
#             torch.nn.Linear(h, 180),
#
#         )
#         # self.r = torch.nn.Parameter(torch.zeros(100,2))
#         # self.r.requires_grad = False
#         # self.l = torch.nn.Parameter(torch.zeros(100,2))
#         # self.l.requires_grad = False
#
#         n_points = 5
#         a = np.array([[(i + 0.5) / n_points, 0.2] for i in range(n_points)])
#         b = np.array([[(i + 0.5) / n_points, 0.8] for i in range(n_points)])
#         self.x = torch.nn.Parameter(torch.tensor(a, dtype=torch.float))
#         self.y = torch.nn.Parameter(torch.tensor(b, dtype=torch.float))
#         self.x.requires_grad = True
#         self.y.requires_grad = False
#
#     def forward(self, pts):
#
#         # return torch.cat((self.x,self.w2),dim=1), torch.cat((self.y,self.w2),dim=1)
#         # self.r[:] = 0
#         # self.l[:] = 0
#         r = torch.zeros(self.n_samples, 2, requires_grad=False)
#         l = torch.zeros(self.n_samples, 2, requires_grad=False)
#         # idx_r = self.w>=1
#         # self.r[idx_r] = -1
#         # idx_l = self.w<0
#         # self.l[idx_l] = 1
#         # x = self.w[0:10]
#         # y = self.w[10:]
#         y = self.w
#
#         # if self.use_mlp:
#         #     y = y.view(1,-1)
#         #     res = self.mlp(y) # green
#         #     y = y + res
#         #     y = y.view(90,2)
#         # y = torch.cat((x,y),dim=0)
#
#         idx_r = y > 1
#         idx_l = y < 0
#         idx_r1 = y == 1
#         idx_l0 = y == 0
#
#         r[idx_r] = -torch.floor(y[idx_r])
#         l[idx_l] = -torch.floor(y[idx_l])
#         y = y + r
#         y = y + l
#
#         y[idx_r1] = 1 - y[idx_r1]
#         y[idx_l0] = 1 - y[idx_l0]
#
#         # print ('after r:',y[idx_r])
#         # print ('after l:',y[idx_l])
#         y = torch.cat((y, self.w2), dim=1)
#         return y
#
#
# class PCF2(torch.nn.Module):
#     def __init__(self, device, nbbins=50, sigma=0.25, npoints=100,
#                  n_rmax=2.5):  # n_rmax=2.5, nbbins=50, sigma=0.02, npoints=1000
#         super(PCF2, self).__init__()
#
#         d = 2 * np.sqrt(1.0 / (2 * np.sqrt(3) * npoints))
#         self.rmax = d
#         self.nbbins = nbbins
#         rs = []
#         area = []
#         stepsize = n_rmax / nbbins
#         for i in range(nbbins):
#             radii = (i + 1) * stepsize * self.rmax
#             rs.append(radii)
#             inner = max(0, radii - 0.5 * self.rmax);
#             outer = radii + 0.5 * self.rmax;
#             area.append(math.pi * (outer * outer - inner * inner))
#
#         self.rmin = rs[0]
#         ra = rs[0]
#         rb = rs[-1]
#         # print (rs)
#         # print (area)
#         print('parameters:', self.rmin, self.rmax, npoints)
#
#         self.rs = torch.FloatTensor(np.array(rs)).to(device)
#         self.area = torch.FloatTensor(np.array(area)).to(device)
#
#         self.sigma = torch.FloatTensor([sigma]).to(device)
#         self.gf = (1.0 / (np.sqrt(math.pi) * self.sigma))
#
#     def gaussianKernel(self, x):
#         return self.gf * torch.exp(-(x * x) / (self.sigma * self.sigma))
#
#     def perimeter_weight(self, x, y):
#
#         full_angle = torch.ones(self.nbbins) * 2 * math.pi
#
#         dx = x
#         dy = y
#         idx = self.rs > dx
#         alpha = torch.acos(dx / self.rs[idx])
#         if self.rs[idx].size(0) > 0:
#             full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))
#
#         dx = 1 - x
#         dy = y
#         idx = self.rs > dx
#         alpha = torch.acos(dx / self.rs[idx])
#         if self.rs[idx].size(0) > 0:
#             full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx));
#
#         dx = y
#         dy = x
#         idx = self.rs > dx
#         alpha = torch.acos(dx / self.rs[idx])
#         if self.rs[idx].size(0) > 0:
#             full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx));
#
#         dx = 1 - y
#         dy = x
#         idx = self.rs > dx
#         alpha = torch.acos(dx / self.rs[idx])
#         if self.rs[idx].size(0) > 0:
#             full_angle[idx] -= torch.min(alpha, torch.atan2(dy, dx)) + torch.min(alpha, torch.atan2((1 - dy), dx))
#
#         return torch.clamp(full_angle / (2 * math.pi), 0, 1)
#
#     def compute_weights(self, gen_pts, relation):
#         # print (len(gen_pts))
#         gen_pts = torch.FloatTensor(np.array(gen_pts))
#         pts = gen_pts[gen_pts[:, 3] == relation]
#         other_weights = []
#         # print(pts.size(0))
#         for i in range(pts.size(0)):
#             p = pts[i:i+1]
#             weights = 1 / self.perimeter_weight(p[:, 0], p[:, 1])
#             # print(weights)
#             other_weights.append(weights)
#         other_weights = torch.stack(other_weights, dim=0)
#         # print (other_weights[0], other_weights.size())
#         return other_weights
#
#     def compute_contribution(self, disk, gen_pts, other_weights, target_pts, id, relation):
#
#         other_weights = other_weights[relation]
#         gen_pts = torch.FloatTensor(np.array(gen_pts))
#         other = gen_pts[gen_pts[:, 3] == relation]
#         cur = gen_pts[gen_pts[:, 3] == id]
#
#         denorm_1 = target_pts[target_pts[:, 3] == id].size(0)
#         denorm_2 = target_pts[target_pts[:, 3] == relation].size(0)
#         denom = denorm_1 * denorm_2
#
#         # print (gen_pts, other)
#         disk = torch.FloatTensor(np.array(disk)).unsqueeze(0)
#         # print (disk.size())
#         disk = disk.repeat(other.size(0), 1)
#
#         # pcf = torch.zeros(self.nbbins)
#         # weights = torch.zeros(self.nbbins)
#         # contribution = torch.zeros(self.nbbins)
#
#         # for i in range(other.size(0)):
#
#         dis = utils.fnorm(disk[:, 0:3], other[:, 0:3], self.rmax)
#         # print (self.rmax)
#         # print ('disk:',disk[:,0:3])
#         # print ('other:',other[:,0:3])
#         # print (dis)
#         # print (other_weights)
#
#         val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - dis.view(-1, 1).repeat(1, self.nbbins))
#
#         weights = 1 / self.perimeter_weight(disk[0, 0], disk[0, 1])
#
#         # print (val.size(), other_weights.size())
#
#         contribution = torch.sum(other_weights * val, 0)
#         # torch.set_printoptions(precision=8)
#         # print (contribution,denom)
#
#         hist = torch.sum(weights * val, 0)
#         pcf = hist / self.area
#
#         contribution = pcf + contribution / self.area
#
#         pcf = pcf / other.size(0)
#         # print (pcf)
#         contribution = contribution / denom
#         # print (contribution,denom)
#         return contribution, pcf
#
#     def compute_error(self, contribution, test_pcf, target_pcf, relation):
#         E = 0
#         eps = 1e-7
#         error_mean = 0
#         error_min = 0
#         error_max = 0
#
#         # pcf0 = target_pcf[0]
#         # pcf1 = target_pcf[1]
#         # target_pcf_mean = torch.mean(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)
#         # target_pcf_max = torch.max(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)[0]
#         # target_pcf_min = torch.min(torch.cat((pcf0[:,1:2], pcf1[:,1:2]),dim=1),dim=1)[0]
#         target_pcf_mean = target_pcf[relation][:, 1]
#         target_pcf_min = target_pcf[relation][:, 2]
#         target_pcf_max = target_pcf[relation][:, 3]
#
#         # print (torch.max((test_pcf - target_pcf_max)/(target_pcf_max+eps)))
#         error_mean = max(((contribution - target_pcf_mean) / (target_pcf_mean + eps)).max(), error_mean)
#         # print ('error_mean: ', error_mean)
#         error_max = max(((test_pcf - target_pcf_max) / (target_pcf_max + eps)).max(), error_max)
#         # print ('error_max: ', error_max)
#         error_min = max(((target_pcf_min - test_pcf) / (target_pcf_min + eps)).max(), error_min)
#         # print (error_min)
#         E = error_mean + max(error_max, error_min);
#         # print (E)
#         return E
#
#     def forward(self, pts, c, t, dimen=3, use_fnorm=False, mean_pcf=False):
#         pcf = torch.zeros(8, 2)
#         hist = torch.zeros(8)
#
#         pcf_lower = torch.ones(self.nbbins) * 1e4
#         pcf_upper = torch.zeros(self.nbbins)
#
#         # if mean_pcf:
#         #     denom = pts.size(0) * pts.size(0)
#         #     pts[0:10,...] = pts[0:10,...].detach()
#         #     for i in range(pts.size(0)):
#
#         #         pts_2 = torch.cat((pts[i+1:,0:dimen],pts[:i,0:dimen]),dim=0)
#         #         pts_1 = pts[i,0:dimen].repeat(pts_2.size(0),1)
#
#         #         if use_fnorm:
#         #             dis = utils.fnorm(pts_1, pts_2, self.rmax)
#         #             val = self.gaussianKernel( self.rs.view(1,-1)/self.rmax - dis.view(-1,1).repeat(1,self.nbbins) )
#
#         #         else:
#         #             dis = utils.euclidean(pts_1, pts_2)
#         #             val = self.gaussianKernel( (self.rs.view(1,-1) - dis.view(-1,1).repeat(1,self.nbbins))/self.rmax )
#
#         #         hist = hist + torch.sum(val,0)
#         #     # cov = 1.0 - ((2.0*self.rs)/math.pi)*2.0 + ((self.rs*self.rs)/math.pi)
#         #     # factor = 2.0*math.pi*self.rs/self.rmax#*cov
#         #     pcf[:,1] = hist / (denom * self.area)
#         #     pcf[:,0] = self.rs / self.rmax
#         #     return pcf
#
#         pts_c = pts[pts[:, 3] == c]
#         pts_t = pts[pts[:, 3] == t]
#         print(c, t)
#         # npoints = (pts_t.size(0)+pts_c.size(0))/2
#         npoints = pts_t.size(0)
#         # print (pts_c.size(0), pts_t.size(0))
#         denom = npoints * npoints
#
#         # for i in range(pts_c.size(0)):
#         #     pts_1 = pts_c[i,0:dimen].repeat(self.nbbins,1)
#         #     dx = pts_1[0:1,0]
#         #     dy = pts_1[0:1,1]
#         # weights = self.perimeter_weight(dx, dy)
#
#         # print (weights)
#
#         for i in range(pts_c.size(0)):
#
#             if t != c:
#                 pts_2 = pts_t[:, 0:dimen]
#                 pts_1 = pts_c[i, 0:dimen].repeat(pts_2.size(0), 1)
#                 pts_2 = pts_2.detach()
#                 denom = pts_c.size(0) * pts_t.size(0)
#                 denorm_1 = pts_c.size(0)
#                 denorm_2 = pts_t.size(0)
#                 # print (pts_1.size(), pts_2.size())
#             else:
#                 pts_2 = torch.cat((pts_c[:i, 0:dimen], pts_c[i + 1:, 0:dimen]), dim=0)
#                 pts_1 = pts_c[i, 0:dimen].repeat(pts_2.size(0), 1)
#                 denorm_1 = npoints
#                 denorm_2 = npoints
#
#             pts_w = pts_1[0, 0:dimen].view(1, -1)  # same
#             weights = 1 / self.perimeter_weight(pts_w[:, 0], pts_w[:, 1])
#
#             # print (weights)
#
#             # if c != t:
#
#             if use_fnorm:
#                 dis = utils.fnorm(pts_1, pts_2, self.rmax)
#                 val = self.gaussianKernel(self.rs.view(1, -1) / self.rmax - dis.view(-1, 1).repeat(1, self.nbbins))
#             else:
#                 dis = utils.euclidean(pts_1, pts_2)
#                 diff = (self.rs.view(1, -1) - dis.view(-1, 1).repeat(1, self.nbbins)) / self.rmax
#
#                 val = self.gaussianKernel(diff)
#
#             # hist = hist + torch.sum(val,0)
#
#             cur_pcf = torch.sum(val, 1)  # weights
#             hist = hist + cur_pcf
#
#             # pcf_lower = torch.min(pcf_lower, cur_pcf/(denorm_2*self.area))
#             # pcf_upper = torch.max(pcf_upper, cur_pcf/(denorm_2*self.area))
#
#         # cov = 1.0 - ((2.0*self.rs)/math.pi)*2.0 + ((self.rs*self.rs)/math.pi)
#         # factor = 2.0*math.pi*self.rs*cov
#         # factor = 1.0/factor
#         # pcf[:,1] *= factor
#         # print (self.area)
#         pcf[:, 1] = hist / denom
#         # pcf[:,0] = self.rs / self.rmax
#
#         # print ('pcf_lower:',pcf_lower)
#         # print ('pcf_upper:',pcf_upper)
#         # pcf = torch.cat((pcf,pcf_lower.view(-1,1),pcf_upper.view(-1,1)),dim=1)
#         # print (pcf.size())
#         return pcf
#

