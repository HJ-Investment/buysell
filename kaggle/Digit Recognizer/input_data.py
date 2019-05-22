import tensorflow as tf
import pandas as pd
import numpy as np
import os

import scipy.misc


# df = pd.read_csv('./kaggle/train.csv', sep=',')
# labels = df.iloc[:, :1]
# images = df.iloc[:, 1:]
# print(labels)

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
    resized_image = tf.image.resize_nearest_neighbor(
        image, [output_height, output_width], align_corners=False)
    resized_image = tf.squeeze(resized_image)
    # resized_image = tf.expand_dims(resized_image, 2)
    # resized_image.set_shape([None, None, 1])
    return resized_image


def read(df):
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))

    def map_fn(element, label_feat):
        # element is a {'c0': int, 'c1': str, 'c2': int} dictionary
        label = element.pop(label_feat)
        # img = list(element.values())
        return (element, label)


    # Split it into features, label tuple
    dataset = dataset.map(lambda elem: map_fn(elem, 'label'))

    # One shot iterator iterates through the (repeated) dataset once

    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()

    # image_raw = tf.reshape(image, [28, 28])

    # images, label_batch = tf.train.batch(
    #     [image_raw, label],
    #     batch_size=32,
    #     num_threads=1,
    #     capacity=64)

    # n_classes = 10
    # label_batch = tf.one_hot(label_batch, depth=n_classes)
    # label_batch = tf.cast(label_batch, dtype=tf.int32)
    # label_batch = tf.reshape(label_batch, [32, n_classes])

    return image, label


def get(df, batch_size):

    def _parse_function(label, img):
        image_raw = tf.reshape(img, [28, 28, 1])
        return label, image_raw

    labels = df['label']
    df.drop(['label'], axis=1, inplace=True)
    imgs = df
    dataset = tf.data.Dataset.from_tensor_slices((labels, imgs))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    label, image = iterator.get_next()
    return image, label

def val(df, batch_size):
    def _parse_function(img):
        image_raw = tf.reshape(img, [28, 28, 1])
        return image_raw

    dataset = tf.data.Dataset.from_tensor_slices(df)
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    image = iterator.get_next()
    return image

def get_resnet(df, batch_size):
    def _parse_function(label, img):
        img = tf.reshape(img, [28, 28, 1])
        image_raw = _fixed_sides_resize(img, 224, 224)
        return label, image_raw

    labels = df['label']
    df.drop(['label'], axis=1, inplace=True)
    imgs = df
    dataset = tf.data.Dataset.from_tensor_slices((labels, imgs))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(10)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    label, image = iterator.get_next()
    return image, label


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    df = pd.read_csv('./kaggle/Digit Recognizer/data/train.csv', sep=',')
    reshaped_image = get_resnet(df, 10)

    for i in range(1):
        # 每次sess.run(reshaped_image)，都会取出一张图片
        imgs, labels = sess.run(reshaped_image)
        k = 0
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        for img in imgs:
            print('cccccccccccccccccccccccccccccccccccccc')
            # np.savetxt("prediction" + str(k) + ".csv", img, delimiter=",")
            scipy.misc.imsave('./kaggle/Digit Recognizer/output/img/test_%d.jpg' % k, img)
            # with open() as f:
            #     f.write(img)
            k += 1
        for label in labels:
            print(label)
        # print(imgs)
        # np.savetxt("prediction" + str(i) + ".csv", img, delimiter=",")
        # print(label)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     df = pd.read_csv('./kaggle/test.csv', sep=',')
#     image_raw = sess.run(tf.reshape(df[5:6], [28, 28]))
#     np.savetxt("prediction.csv", image_raw, delimiter=",")