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

    # image_raw = tf.reshape(image, [28, 28])

    return image, label

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    reshaped_image = read(df)

    for i in range(5):
        # 每次sess.run(reshaped_image)，都会取出一张图片
        img, label = sess.run(reshaped_image)
        print(img)
        # np.savetxt("prediction" + str(i) + ".csv", img.toarray(), delimiter=",")
        print(label)

# np.savetxt("prediction.csv", image_raw, delimiter=",")