
import os
import cv2
import numpy as np
np.random.seed(10)

# 原始图片150张
# 每张图片随机选250个位置进行crop, 论文里面是2500个patch, 实验用不了那么多数据
num_patches_each = 250 # 250

root_dir = '/home/wuyanxue/Data/StandardTestImages/dataset/'

origin_dir = os.path.join(root_dir, 'origin')

image_names = os.listdir(origin_dir)

first_image = cv2.imread(os.path.join(origin_dir, image_names[0]), cv2.IMREAD_UNCHANGED)
height, width = first_image.shape[:2]

with open(os.path.join(root_dir, 'train.txt'), 'r') as f:
    contents = f.readlines()

f_contents_train = contents[:num_patches_each]

with open(os.path.join(root_dir, 'test.txt'), 'r') as f:
    contents = f.readlines()

f_contents_test = contents[:num_patches_each]


origin_image = np.zeros((height, width))
image_num = np.zeros((height, width))
noise_image = np.zeros((height, width))
darken_image = np.zeros((height, width))
noise_darken_image = np.zeros((height, width))

rand_xx = np.random.randint(low=0, high=width-17, size=num_patches_each)
rand_yy = np.random.randint(low=0, high=height-17, size=num_patches_each)


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

count_train = 0
count_test = 0

for i in range(int(num_patches_each * 2 / 3)):

    x = f_contents_train[count_train]
    y = f_contents_test[count_test]

    xx = x.strip().split(' ')
    origin_patch = [float(y) for y in xx[:17 * 17]]
    origin_patch = np.reshape(origin_patch, newshape=(17, 17))

    _ = add_gaussian_noise(origin_patch).reshape(-1)
    _ = darkened(origin_patch).reshape(-1)
    _ = add_gaussian_noise_darkened(origin_patch).reshape(-1)

    # train and test data ratio 2:1
    if np.random.randint(0, 3) == 0:
        x = y
        count_test += 1
    else:
        count_train += 1

    xx = x.strip().split(' ')
    origin_patch = [float(y) for y in xx[:17*17]]
    origin_patch = np.reshape(origin_patch, newshape=(17, 17))

    noise_patch = [float(y) for y in xx[17*17:17*17*2]]
    noise_patch = np.reshape(noise_patch, newshape=(17, 17))

    darken_patch = [float(y) for y in xx[17*17*2:17*17*3]]
    darken_patch = np.reshape(darken_patch, newshape=(17, 17))

    noise_darkened_patch = [float(y) for y in xx[17*17*3:17*17*4]]
    noise_darkened_patch = np.reshape(noise_darkened_patch, newshape=(17, 17))

    # reconstruct from patches
    origin_image[rand_yy[i]:rand_yy[i]+17, rand_xx[i]:rand_xx[i]+17] += origin_patch

    noise_image[rand_yy[i]:rand_yy[i]+17, rand_xx[i]:rand_xx[i]+17] += noise_patch

    darken_image[rand_yy[i]:rand_yy[i]+17, rand_xx[i]:rand_xx[i]+17] += darken_patch

    noise_darken_image[rand_yy[i]:rand_yy[i]+17, rand_xx[i]:rand_xx[i]+17] += noise_darkened_patch

    image_num[rand_yy[i]:rand_yy[i]+17, rand_xx[i]:rand_xx[i]+17] += 1

image_num[image_num == 0] = 1


origin_image = np.asarray(origin_image * 255 / image_num, dtype=np.uint8)
origin_image_rgb = np.dstack([origin_image, origin_image, origin_image])

noise_image = np.asarray(noise_image * 255 / image_num, dtype=np.uint8)
noise_image_rgb = np.dstack([noise_image, noise_image, noise_image])

darken_image = np.asarray(darken_image * 255 / image_num, dtype=np.uint8)
darken_image_rgb = np.dstack([darken_image, darken_image, darken_image])

noise_darken_image = np.asarray(noise_darken_image * 255 / image_num, dtype=np.uint8)
noise_darken_image_rgb = np.dstack([noise_darken_image, noise_darken_image, noise_darken_image])

if not os.path.exists('./test_result/test_input'):
    os.mkdir('./test_result/test_input')

cv2.imwrite('./test_result/test_input/origin_image_rgb.jpg', origin_image_rgb)
cv2.imwrite('./test_result/test_input/noise_image_rgb.jpg', noise_image_rgb)
cv2.imwrite('./test_result/test_input/darken_image_rgb.jpg', darken_image_rgb)
cv2.imwrite('./test_result/test_input/noise_darken_image_rgb.jpg', noise_darken_image_rgb)

