
import os
import cv2
import numpy as np
np.random.seed(10)

# 原始图片150张
# 每张图片随机选250个位置进行crop, 论文里面是2500个patch, 实验用不了那么多数据
num_patches_each = 2500 # 250

root_dir = '/home/wuyanxue/Data/StandardTestImages/dataset/'

origin_dir = os.path.join(root_dir, 'origin')

image_names = os.listdir(origin_dir)


def add_gaussian_noise(im_gray):
    height, width = im_gray.shape[:2]
    mean = 0
    var = 0.1
    sigma = (np.random.uniform() * var**2)**0.5
    gauss = np.random.normal(mean, sigma, (height, width))
    _im_gray_noise = im_gray + gauss
    _im_gray_noise = np.clip(_im_gray_noise, 0, 1)
    return _im_gray_noise


def darkened(im_gray):
    gamma = np.random.uniform(2, 5)
    return im_gray ** gamma


def add_gaussian_noise_darkened(im_gray):
    _im_gray_darkened = darkened(im_gray)
    return add_gaussian_noise(_im_gray_darkened)


out_train_file = open(os.path.join(root_dir, 'train_2500patch_per_image.txt'), 'w')
out_test_file = open(os.path.join(root_dir, 'test_2500patch_per_image.txt'), 'w')

for i, im_name in enumerate(image_names):
    print('{}-th image done...'.format(i))
    im_gray = cv2.imread(os.path.join(origin_dir, im_name), cv2.IMREAD_UNCHANGED)
    im_gray = im_gray / 255.

    height, width = im_gray.shape[:2]
    # [0, 1] float type
    # crop patches
    rand_xx = np.random.randint(low=0, high=width-17, size=num_patches_each)
    rand_yy = np.random.randint(low=0, high=height-17, size=num_patches_each)

    # origin_patches noise_only darkened_only noise_darkened
    for xx, yy in zip(rand_xx, rand_yy):

        origin_patch = im_gray[yy:yy+17, xx:xx+17]
        noise_only_patch = add_gaussian_noise(origin_patch).reshape(-1)
        darkened_only_patch = darkened(origin_patch).reshape(-1)
        noise_darkened_patch = add_gaussian_noise_darkened(origin_patch).reshape(-1)
        origin_patch = origin_patch.reshape(-1)

        a_line = []
        a_line.extend(origin_patch)
        a_line.extend(noise_only_patch)
        a_line.extend(darkened_only_patch)
        a_line.extend(noise_darkened_patch)
        a_line_str = list(map(str, a_line))
        a_line_str = ' '.join(a_line_str) + '\n'

        # train and test data ratio 2:1
        if np.random.randint(0, 3) == 0:
            out_test_file.write(a_line_str)
        else:
            out_train_file.write(a_line_str)

out_train_file.close()
out_test_file.close()









