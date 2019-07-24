import os
import struct
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image

import tensorflow as tf
import cv2

aph = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

def load_emnist_image(path, imagefilename, labelfilename, type = 'train', batch_size=64):

    image_full_name = os.path.join(path, imagefilename)
    label_full_name = os.path.join(path, labelfilename)
    fp1 = open(image_full_name, 'rb')
    fp2 = open(label_full_name, 'rb')
    buf1 = fp1.read()
    buf2 = fp2.read()

    # 处理labels
    index = 0;
    magic, num = struct.unpack_from('>II', buf2, index)
    index += struct.calcsize('>II')
    labels = []

    for i in range(num):
        l = int(struct.unpack_from('>B',buf2, index)[0])
        labels.append(l)
        index += struct.calcsize('>B')
    

    # 处理images
    index = 0;
    magic_image, num_image, rows_image, cols_image = struct.unpack_from('>IIII', buf1, index)
    magic, num = struct.unpack_from('>II', buf1, index)
    index += struct.calcsize('>IIII')
    images = []

    for image in range(0, num):
        im = struct.unpack_from('>784B', buf1, index)
        index += struct.calcsize('>784B')
        im = np.array(im, dtype = 'uint8')
        im = im.reshape(28, 28)
        im_rot90 = np.rot90(im, -1)
        im_mirror = np.fliplr(im_rot90)
        # im_mirror = Image.fromarray(im_mirror)
        images.append(im_mirror)
        # if (type == 'train'):
        #     print(image)
        #     im_mirror.save("/home/rinz/Documents/buysell/read_picture/handwriting/datasets/rawdata/aa/train_{a}_{b}.png".format(a=aph[int(labels[image])-1], b=image), 'png')
        # if (type == 'test'):
        #     im_mirror.save('/home/rinz/Documents/buysell/read_picture/handwriting/datasets/rawdata/bb/test_%s.png' %image, 'png')

    # 构建dataset
    def _fixed_sides_resize(image, output_height, output_width):
        """Resize images by fixed sides.

        Args:
            image: A 3-D image `Tensor`.
            output_height: The height of the image after preprocessing.
            output_width: The width of the image after preprocessing.

        Returns:
            resized_image: A 3-D tensor containing the resized image.
        """
        output_height = tf.convert_to_tensor(output_height, dtype=tf.int32)
        output_width = tf.convert_to_tensor(output_width, dtype=tf.int32)

        image = tf.expand_dims(image, 0)
        # resized_image = tf.image.resize_nearest_neighbor(
        #     image, [output_height, output_width], align_corners=False)  # 返回[batch, height, width, channels]
        resized_image = tf.image.resize_images(
            image, [output_height, output_width], method=0)
        resized_image = tf.squeeze(resized_image, 0)  # 去掉batch，留下[224, 224, 1]
        resized_image = tf.concat([resized_image, resized_image, resized_image], -1)  # 单通道叠到3通道
        # resized_image = tf.expand_dims(resized_image, 2)
        # resized_image.set_shape([None, None, 1])
        return resized_image

    def _parse_function(image, label):
        img = tf.reshape(image, [28, 28, 1])
        image_raw = _fixed_sides_resize(img, 224, 224)
        return tf.to_float(image_raw), label-1

    images_array = np.array(images)
    dataset = tf.data.Dataset.from_tensor_slices((images_array, labels))
    dataset = dataset.map(_parse_function)
    if type == 'train':
        dataset = dataset.repeat(10)
    # dataset = dataset.batch(64)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    return dataset
#     iterator = dataset.make_one_shot_iterator()
#     image, label = iterator.get_next()
#     return image, label


# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())

#     path = '/tf/handwriting/datasets/'
#     train_images = 'emnist-letters-train-images-idx3-ubyte'
#     train_labels = 'emnist-letters-train-labels-idx1-ubyte'

#     reshaped_image = load_emnist_image(path, train_images, train_labels)

#     for i in range(1):
#         # 每次sess.run(reshaped_image)，都会取出一张图片
#         imgs, labels = sess.run(reshaped_image)
#         # print(labels)
#         # print('--------------------------------')
#         # labels_max = tf.reduce_max(labels)
#         # if(sess.run(labels_max) > 26):
#         #     print('bad')
#         # print('--------------------------------')
#         index=0
#         print(len(imgs))
#         print("------start-------")
#         for img in imgs:
#             print(labels[index])
#             # print(img)
#             cv2.imwrite("/tf/handwriting/datasets/rawdata/raw_test/train_{a}_{b}.png".format(
#                                                                 a = aph[labels[index]], b = index), img)
#             index += 1