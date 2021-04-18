import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


def render(categories_elem, elements):
    print('render')
    res = 512
    vis_img = np.ones((res, res, 4))
    for k in categories_elem.keys():  # assume number of classes is equal to number of elements type
        for e in categories_elem[k]:
            ratio = 0.5 / e[0, 2]
            im = elements[k]
            im = im.resize((int(res / ratio), int(res / ratio)), resample=Image.LANCZOS)
            im = np.asarray(im)
            # plt.figure(1)
            # plt.imshow(im)
            # plt.show()
            size_x, size_y = im.shape[0], im.shape[1]

            if size_x % 2 == 0:
                left, right = size_x // 2, size_x // 2
            else:
                left, right = size_x // 2, size_x // 2 + 1
            if size_y % 2 == 0:
                up, down = size_y // 2, size_y // 2
            else:
                up, down = size_y // 2, size_y // 2 + 1

            # print(e.shape)

            new_x = int(e[0, 0] * res)
            new_y = int(e[0, 1] * res)
            # print(new_x, new_y)
            border = 0
            start_x = max(new_x - left - border, 0)
            end_x = min(new_x + right + border, res)
            start_y = max(new_y - up - border, 0)
            end_y = min(new_y + down + border, res)
            # print(start_x, end_x, start_y, end_y)
            if new_y + down >= res > new_x + right:
                img_crop = im[0:end_x - start_x, new_y + down - res:].copy()
            elif new_x + right >= res > new_y + down:
                img_crop = im[new_x + right - res:, 0:end_y - start_y].copy()
            elif new_y + down >= res and new_x + right >= res:
                img_crop = im[new_x + right - res:, new_y + down - res:].copy()
            else:
                img_crop = im[0:end_x - start_x, 0:end_y - start_y].copy()

            # plt.figure(1)
            # plt.imshow(img_crop)
            # plt.show()
            # print(img_crop)
            vis_img[res - end_x:res - start_x, res - end_y:res - start_y, :] += img_crop
            # print(vis_img)
    plt.figure(1)
    plt.imshow(np.mean(vis_img, -1))
    plt.show()
