import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean


def render(categories_elem, categories_radii_ratio, elements, save_filename):
    print('render')
    res = 512
    vis_img = np.ones((res, res, 3))
    fig, ax = plt.subplots()
    # for k in categories_elem.keys():  # assume number of classes is equal to number of elements type
    for k in range(len(categories_elem)):  # assume number of classes is equal to number of elements type
        for l in range(len(categories_elem[k])):
            e = categories_elem[k][l].detach().cpu().numpy()
            # ratio = 0.5 / e[0, 2]
            ratio = categories_radii_ratio[k][l]
            # print(ratio, categories_radii_ratio[k][l])
            im = elements[k]
            # im = im.resize((int(res / ratio), int(res / ratio)), resample=Image.LANCZOS)
            im = np.asarray(im) / 255.0

            im = np.where(im == [0, 0, 0], 1, im)

            # im = resize(im, (int(res / ratio), int(res / ratio), 3), order=0, preserve_range=True)
            # im = resize(im, (int(res / ratio), int(res / ratio), 3))

            im = Image.fromarray((im * 255).astype('uint8'), 'RGB')
            # im_rotate = im.rotate(45, expand=True)
            im_rotate = im.rotate(45, expand=1, fillcolor=(255, 255, 255))
            w, h = im.size
            # im_rotate = im_rotate.resize((w, h), resample=Image.LANCZOS)
            # print(im, im_rotate)
            # new_width, new_height = im_rotate.size
            # im_pad = Image.new(im.mode, (new_width, new_height), (255, 255, 255))
            # im_pad.paste(im, (0, 0))
            # plt.figure(1)
            # plt.subplot(121)
            # plt.imshow(im_pad)
            # plt.subplot(122)
            # plt.imshow(im_rotate)
            # plt.show()
            im = im.resize((int(res / ratio), int(res / ratio)), resample=Image.LANCZOS)
            im = np.asarray(im) / 255

            # print(im.max(), im.min())

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

            new_x = int((1 - e[0, 1]) * res)    # need to flip x, y because images store value in this way
            new_y = int((e[0, 0]) * res)
            # print(new_x, new_y)
            border = 0
            start_x = max(new_x - left - border, 0)
            end_x = min(new_x + right + border, res)
            start_y = max(new_y - up - border, 0)
            end_y = min(new_y + down + border, res)
            # print('start end:', start_x, end_x, start_y, end_y)
            img_crop = im.copy()
            if new_y - up < 0:
                # print('(new_y - up):', new_y, up, 0 - (new_y - up))
                img_crop = img_crop[:, 0 - (new_y - up):, :]
            if new_y + down >= res:
                img_crop = img_crop[:, 0:size_y - (new_y + down - res), :]
            if new_x - left < 0:
                img_crop = img_crop[0 - (new_x - left):, :, :]
            if new_x + right >= res:
                img_crop = img_crop[0:size_x-(new_x+right-res), :, :]
            # if new_y + down >= res > new_x + right:
            #     img_crop = im[0:end_x - start_x, 0:new_y + down - res, :].copy()
            # elif new_x + right >= res > new_y + down:
            #     img_crop = im[0:new_x + right - res, 0:end_y - start_y].copy()
            # elif new_y + down >= res and new_x + right >= res:
            #     img_crop = im[0:new_x + right - res:, new_y + down - res:].copy()
            # else:
            #     img_crop = im[0:end_x - start_x, 0:end_y - start_y].copy()

            # plt.figure(1)
            # plt.imshow(img_crop)
            # plt.show()
            # print(img_crop)
            # vis_img[res - end_x:res - start_x, res - end_y:res - start_y, :] += img_crop
            # print(res - end_x, res - start_x, res - end_y, res - start_y, img_crop.shape)
            # vis_img[res - end_x:res - start_x, res - end_y:res - start_y, :] += img_crop
            # vis_img[res - end_y:res - start_y, res - end_x:res - start_x, :] += img_crop
            vis_img[start_x:end_x, start_y:end_y, :] += img_crop

            # print(vis_img)
    # vis_img[vis_img != 1] = vis_img[vis_img != 1] / 255
    Image.fromarray((vis_img * 255).astype('uint8'), 'RGB').save(save_filename+'.png')

    image = Image.open(save_filename+'.png')
    draw = ImageDraw.Draw(image)
    for k in range(len(categories_elem)):  # assume number of classes is equal to number of elements type
        for l in range(len(categories_elem[k])):
            e = categories_elem[k][l]
            x = int((e[0, 0]) * res)
            y = int((1 - e[0, 1]) * res)
            r = int((e[0, 2]) * res)
            # print(x, y, r)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=None, outline=(255, 0, 0))
            for m in range(1, e.shape[0]):
                x = int((e[m, 0]) * res)
                y = int((1 - e[m, 1]) * res)
                r = int((e[m, 2]) * res)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=None, outline=(0, 255, 0))
    image.save(save_filename+'_circle.png')
    # ax.imshow(vis_img)
    # plt.show()
    # plt.figure(1)
    # plt.imshow(vis_img)
    # plt.savefig('results/first/render')
    # plt.clf()
