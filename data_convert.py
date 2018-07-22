# coding:utf-8
from __future__ import absolute_import
import argparse
import os
import logging
from tfrecord import main
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tensorflow-data-dir', default='F:\\Code\\buysell\\data')
    parser.add_argument('--train-shards', default=2, type=int)
    parser.add_argument('--validation-shards', default=2, type=int)
    parser.add_argument('--num-threads', default=2, type=int)
    parser.add_argument('--dataset-name', default='macd', type=str)
    return parser.parse_args()

def decodeTFRecords(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/class/label': tf.FixedLenFeature([], tf.int64),
                                       })  # 将image数据和label取出来
    # img = features['img_raw']
    # img = tf.image.decode_png(img, channels=3)  # 这里，也可以解码为 1 通道
    # img = tf.image.resize_images(img, [128, 128])

    # label = tf.cast(features['image/class/label'], tf.int64)  # 在流中抛出label张量
    return [features['image/class/label']]

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.tensorflow_dir = args.tensorflow_data_dir
    args.train_directory = os.path.join(args.tensorflow_dir, 'train')
    args.validation_directory = os.path.join(args.tensorflow_dir, 'validation')
    args.output_directory = args.tensorflow_dir
    args.labels_file = os.path.join(args.tensorflow_dir, 'label.txt')
    if os.path.exists(args.labels_file) is False:
        logging.warning('Can\'t find label.txt. Now create it.')
        all_entries = os.listdir(args.train_directory)
        dirnames = []
        for entry in all_entries:
            if os.path.isdir(os.path.join(args.train_directory, entry)):
                dirnames.append(entry)
        with open(args.labels_file, 'w') as f:
            for dirname in dirnames:
                f.write(dirname + '\n')
    main(args)


# with tf.Session() as sess:
#     filename_queue = tf.train.string_input_producer(['F:\\Code\\buysell\\data\\tfrecords\\macd_train_00000-of-00002.tfrecord'])
#     label_fun = decodeTFRecords(filename_queue)
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
#     for i in range(5):
#         # 每次sess.run(reshaped_image)，都会取出一张图片
#         label = sess.run(label_fun)
#         # img = np.fromstring(img, np.uint8)
#         # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#         # img = tf.cast(img, tf.float32)
#         # print(img.dtype)
#         # print(sess.run(tf.shape(img)))
#         # 将图片保存
#         # ds = pd.DataFrame(img.tolist())
#         # ds.to_csv('F:\Code\\buysell\data\pic_data\\raw_test\\%d.csv' % i)
#         # print(img)
#         # cv2.imwrite('F:\Code\\buysell\data\pic_data\\raw_test\\%d.png' % i, img)
#         print(label)