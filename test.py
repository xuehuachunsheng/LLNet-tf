# 测试tf加噪声

import tensorflow as tf

path1 = open('/Users/wuyanxue/Desktop/test1.jpg', 'rb').read()
path2 = open('/Users/wuyanxue/Desktop/test2.jpg', 'rb').read()

im1 = tf.image.decode_jpeg(path1)
im2 = tf.image.decode_jpeg(path2)

im2_resize = tf.image.resize_bilinear([im2], tf.shape(im1)[:2])[0]

im2_resize = tf.clip_by_value(im2_resize, 0, 255)
im2_resize = tf.cast(im2_resize, dtype=tf.uint8)

ssim = tf.image.ssim(im1[None,:,:,0][...,None], im2_resize[None, :,:,0][...,None], max_val=255)
psnr = tf.image.psnr(im1[None,:,:,0][...,None], im2_resize[None, :,:,0][...,None], max_val=255)

s = tf.Session()

print(s.run([ssim, psnr]))



