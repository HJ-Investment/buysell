import tensorflow as tf
import pandas as pd
import numpy as np
import os


df = pd.read_csv('./train.csv', sep=',')
# labels = df.iloc[:, :1]
# images = df.iloc[:, 1:]
# print(labels)


def read(df):
    dataset = tf.data.Dataset.from_tensor_slices(dict(df))

    def map_fn(element, label_feat):
        # element is a {'c0': int, 'c1': str, 'c2': int} dictionary
        label = element.pop(label_feat)
        return (element, label)


    # Split it into features, label tuple
    dataset = dataset.map(lambda elem: map_fn(elem, 'label'))

    # One shot iterator iterates through the (repeated) dataset once

    iterator = dataset.make_one_shot_iterator()
    image, label = iterator.get_next()

    image_raw = tf.reshape(list(image.values()), [28, 28])

    images, label_batch = tf.train.batch(
        [image_raw, label],
        batch_size=32,
        num_threads=64,
        capacity=2000)

    n_classes = 10
    label_batch = tf.one_hot(label_batch, depth=n_classes)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [32, n_classes])

    return images, label_batch

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    reshaped_image = read(df)

    for i in range(1):
        # 每次sess.run(reshaped_image)，都会取出一张图片
        imgs, labels = sess.run(reshaped_image)
        k = 0
        for img in imgs:
            np.savetxt("prediction" + str(k) + ".csv", img, delimiter=",")
            k += 1
        for label in labels:
            print(label)
        # np.savetxt("prediction" + str(i) + ".csv", img, delimiter=",")
        # print(label)

# np.savetxt("prediction.csv", image_raw, delimiter=",")