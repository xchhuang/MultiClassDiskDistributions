import numpy as np
# import torch
import matplotlib.pyplot as plt
import math
import glob


def read_pcf(scene_name):
    files = glob.glob('outputs/'+scene_name+'_pcf_*.txt')
    pcfs = []
    ids = []
    for file in files:
        # print(file)
        num = file.split('/')[-1].split('.')[0].split('_')
        # print(num[-2:])
        ids.append('_'.join(num[-2:]))
        with open(file) as f:
            lines = f.readlines()
            xx, yy = lines[0], lines[1]
            xx = xx.strip().split(' ')
            xx = [float(x) for x in xx]
            yy = yy.strip().split(' ')
            yy = [float(y) for y in yy]
        pcfs.append([xx, yy])
    pcfs = np.array(pcfs)
    # print(pcfs.shape)
    return pcfs, ids


def read_txt(path='build/init_pts_final.txt', start=1, normalize=True):
    pts = []
    cla = []
    with open(path) as f:
        lines = f.readlines()
        lines = lines[start:]
        for line in lines:
            line = line.strip().split()
#            if normalize:
            c, x, y, r = line
            x = float(x) / 10000
            y = float(y) / 10000
            c = int(c)
#            else:
#                x, y, c, r = line
#                x = float(x)
#                y = float(y)
#                c = int(c)
            pts.append([x, y])
            cla.append(c)

    pts = np.array(pts)
    cla = np.array(cla)
    return cla, pts


def read_all_txt(path='outputs'):
    files = glob.glob(path+'/*.txt')
    for file in files:
        scene_name = file.split('/')[-1].split('.')[0].split('_')
        scene_name = '_'.join(scene_name[:-1])
        print(scene_name)
        init_cla, init_pts = read_txt(path='outputs/'+scene_name+'_init.txt', start=1, normalize=False)
        tar_cla, tar_pts = read_txt(path='examples/'+scene_name+'.txt', start=1, normalize=True)
        plt.figure(1)
        plt.subplot(121)
        plt.title('Synthesized')
        plt.scatter(init_pts[:, 0], init_pts[:, 1], c=init_cla, s=5)
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.subplot(122)
        plt.title('Input Exemplar')
        plt.scatter(tar_pts[:, 0], tar_pts[:, 1], c=tar_cla, s=5)
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.savefig('outputs/'+scene_name)
        plt.clf()


def plot_txt(scene_name):
    init_cla, init_pts = read_txt(path='outputs/'+scene_name+'_init.txt', start=1, normalize=False)
    tar_cla, tar_pts = read_txt(path='examples/'+scene_name+'.txt', start=1, normalize=True)

    plt.figure(1)
    plt.subplot(121)
    plt.title('Synthesized')
    plt.scatter(init_pts[:, 0], init_pts[:, 1], c=init_cla, s=5)
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.subplot(122)
    plt.title('Input Exemplar')
    plt.scatter(tar_pts[:, 0], tar_pts[:, 1], c=tar_cla, s=5)
    plt.axis('equal')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig('outputs/'+scene_name)
    plt.clf()

    plt.figure(1)
    pcfs, ids = read_pcf(scene_name)
    for i in range(pcfs.shape[0]):
        pcf = pcfs[i]
#        print(pcf.shape)
        plt.plot(pcf[0,:], pcf[1,:])
#    plt.legend(ids)
#    plt.ylim([0, 10])
    plt.savefig('outputs/'+scene_name+'_pcf')
    plt.clf()


def main():
    # read_all_txt()
#    plot_txt('praise_the_sun')
    # plot_txt('zerg_rush')
    plot_txt('forest')
#    plot_txt('constrained')
#    plot_txt('constrained_overlap')


if __name__ == '__main__':
    main()





