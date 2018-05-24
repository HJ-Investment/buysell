# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd = 'D:/train_data/'
classes = {'-1', '0', '1'}  # 人为设定3类
# stocks_list = {'601328','600999', '601628', '600016', '601985', '600019', '600028', '601668', '601878', '601688', '601669', '601390', '601211', '601006', '601989', '601988', '600104', '600518', '601398', '600958', '600309', '601857','600030', '601318', '600837', '600036', '600050', '600547', '601766', '601166', '601601', '601229', '600919', '601169', '600029', '601800', '600000', '600887', '600340', '601881', '601088','603993', '600519', '601336', '601186', '600606', '601818', '600048', '600111', '601288' }
stocks_list = {'601878'}
writer = tf.python_io.TFRecordWriter("up_or_down_train.tfrecords")  # 要生成的文件

for symbol in enumerate(stocks_list):

    writer = tf.python_io.TFRecordWriter("up_or_down_train.tfrecords")  # 要生成的文件
    class_path = cwd + str(symbol) + '/'

    df = pd.read_csv(class_path + str(symbol) + '_final.csv', sep=',')
    labels = df['day_five']
    labels_list = labels.tolist()

    for i, img_name in os.listdir(class_path):
        img_path = class_path + img_name + '.png'# 每一个图片的地址
        print(img_path)
        # img = Image.open(img_path)
        # img = img.resize((128, 128))
        # img_raw = img.tobytes()  # 将图片转化为二进制格式
        img_read = cv2.imread(img_path, 0)
        img = cv2.resize(img_read, (128, 128))
        img_encode = cv2.imencode('.png', img)
        data_encode = np.array(img_encode)
        img_raw = data_encode.tostring()


        print(labels_list[i])

        # example = tf.train.Example(features=tf.train.Features(feature={
        #     "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels_list[i]])),
        #     'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
        # }))  # example对象对label和image数据进行封装
        # writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()


def read_and_decode(filename):  # up_or_down_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来

    img_decode = tf.decode_raw(features['img_raw'], tf.uint8)
    img_decode = tf.reshape(img_decode, [128, 128, 3])  # reshape为128*128的3通道图片
    img_decode = tf.cast(img_decode, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img_decode, label
