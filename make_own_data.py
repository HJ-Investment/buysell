# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc


classes = {'-1', '0', '1'}  # 人为设定3类
stocks_list = {'601328','600999', '601628', '600016', '601985', '600019', '600028', '601668', '601878', '601688', '601669', '601390', '601211', '601006', '601989', '601988', '600104', '600518', '601398', '600958', '600309', '601857','600030', '601318', '600837', '600036', '600050', '600547', '601766', '601166', '601601', '601229', '600919', '601169', '600029', '601800', '600000', '600887', '600340', '601881', '601088','603993', '600519', '601336', '601186', '600606', '601818', '600048', '600111', '601288' }
# stocks_list = {'601878'}
# writer = tf.python_io.TFRecordWriter("up_or_down_train.tfrecords")  # 要生成的文件

def encode_and_write(stocks_list):

    cwd = 'F:\Code\\buysell\data\pic_data\\'
    

    for _, symbol in enumerate(stocks_list):
        writer = tf.python_io.TFRecordWriter("F:\Code\\buysell\data\pic_data\\macd_train_" + symbol + ".tfrecords")  # 要生成的文件

        df = pd.read_csv(cwd + 'datacsv\\' + str(symbol) + '_final.csv', sep=',')
        labels = df['day_five']
        labels_list = labels.tolist()
        class_path = cwd + 'macd_pic\\' + str(symbol) + '\\'
        file_list = os.listdir(class_path)
        file_list_sorted = sorted(file_list,key= lambda x:int(x[:-4]))
        for i, img_name in enumerate(file_list_sorted):
            img_path = class_path + img_name # 每一个图片的地址
            print(img_path)
            # img = Image.open(img_path)
            # img = img.resize((128, 128))
            # img_raw = img.tobytes()  # 将图片转化为二进制格式
            img_read = cv2.imread(img_path, 0)
            img = cv2.resize(img_read, (128, 128))
            img_encode = cv2.imencode('.png', img)[1]
            data_encode = np.array(img_encode)
            img_raw = data_encode.tostring()
            # print(img_raw)


            print(labels_list[i])

            example = tf.train.Example(features=tf.train.Features(feature={
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[labels_list[i]])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))  # example对象对label和image数据进行封装
            writer.write(example.SerializeToString())  # 序列化为字符串
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
    # img_decode = tf.reshape(img_decode, [128, 128, 2])  # reshape为128*128的3通道图片
    # img_decode = tf.cast(img_decode, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img_decode, label

encode_and_write(stocks_list)


# with tf.Session() as sess:
#     reshaped_image = read_and_decode('F:\Code\\buysell\data\pic_data\\macd_train_601878.tfrecords')
#     # 这一步start_queue_runner很重要。
#     # 我们之前有filename_queue = tf.train.string_input_producer(filenames)
#     # 这个queue必须通过start_queue_runners才能启动
#     # 缺少start_queue_runners程序将不能执行
#     threads = tf.train.start_queue_runners(sess=sess)
#     # 变量初始化
#     sess.run(tf.global_variables_initializer())

#     if not os.path.exists('F:\Code\\buysell\data\pic_data\\raw_test'):
#         os.makedirs('F:\Code\\buysell\data\pic_data\\raw_test')
#     # 保存30张图片
#     for i in range(30):
#         # 每次sess.run(reshaped_image)，都会取出一张图片
#         img_str, label = sess.run(reshaped_image)
#         # 将图片保存
#         image = np.fromstring(img_str, np.uint8)  
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#         ds = pd.DataFrame(image.tolist())
#         # ds.to_csv('F:\Code\\buysell\data\pic_data\\raw_test\\%d.csv' % i)
#         # print(image)   
#         cv2.imwrite('F:\Code\\buysell\data\pic_data\\raw_test\\%d.png' % i, image)
#         print(label)
