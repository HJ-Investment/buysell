# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

# image size of 128 x 128. If one alters this number, then the entire model
# architecture will change and any model would need to be retrained.
IMAGE_SIZE = 128

# Global constants describing the MACD/KDJ data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000

cwd = 'F:\Code\\buysell\data\pic_data\\'



classes = {'-1', '0', '1'}  # 人为设定3类
stocks_list = {'601328','600999', '601628', '600016', '601985', '600019', '600028', '601668', '601878', '601688', '601669', '601390', '601211', '601006', '601989', '601988', '600104', '600518', '601398', '600958', '600309', '601857','600030', '601318', '600837', '600036', '600050', '600547', '601766', '601166', '601601', '601229', '600919', '601169', '600029', '601800', '600000', '600887', '600340', '601881', '601088','603993', '600519', '601336', '601186', '600606', '601818', '600048', '600111', '601288' }
# stocks_list = {'601878'}
# writer = tf.python_io.TFRecordWriter("up_or_down_train.tfrecords")  # 要生成的文件

def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def int64_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[values]))

def encode_and_write(stocks_list, dataset_dir, tfrecord_name):
    # writer = tf.python_io.TFRecordWriter(pwd + "tfrecords\\macd_train.tfrecords")  # 要生成的文件
    tfrecord_file = cwd+tfrecord_name
    with tf.python_io.TFRecordWriter(tfrecord_file) as writer:
      for _, symbol in enumerate(stocks_list):
          df = pd.read_csv(cwd + 'datacsv\\' + str(symbol) + '_final.csv', sep=',')
          labels = df['day_five']
          labels_list = labels.tolist()
          class_path = cwd + dataset_dir + str(symbol) + '\\'
          file_list = os.listdir(class_path)
          file_list_sorted = sorted(file_list,key= lambda x:int(x[:-4]))
          for i, img_name in enumerate(file_list_sorted):
              # 有几点需要注意：
              # 1.打包 tfrecord 的时候，千万不要使用 Image.open() 或者 matplotlib.image.imread() 等方式读取。
              # 1张小于10kb的png图片，前者（Image.open) 打开后，生成的对象100+kb, 后者直接生成 numpy 数组，大概是原图片的几百倍大小。
              # 所以应该直接使用 tf.gfile.FastGFile() 方式读入图片。
              # 2.从 tfrecord 中取数据的时候，再用 tf.image.decode_png() 对图片进行解码。
              # 3.不要随便使用 tf.image.resize_image_with_crop_or_pad 等函数，可以直接使用 tf.reshape()。前者速度极慢。
              # 4.如果有固态硬盘的话，图片数据一定要放在固态硬盘中进行读取，速度能高几十倍几十倍几十倍！生成的 tfrecord 文件就无所谓了，找个机械盘放着就行。
              img_path = class_path + img_name # 每一个图片的地址
              print(img_path)
              # # img = Image.open(img_path)
              # # img = img.resize((128, 128))
              # # img_raw = img.tobytes()  # 将图片转化为二进制格式
              # img_read = cv2.imread(img_path, 0)
              # img = cv2.resize(img_read, (128, 128))
              # img_encode = cv2.imencode('.png', img)[1]
              # data_encode = np.array(img_encode)
              # img_raw = data_encode.tostring()
              img_raw = tf.gfile.FastGFile(img_path, 'rb').read()  # 读入图片
              # print(img_raw)


              print(labels_list[i])

              example = tf.train.Example(features=tf.train.Features(feature={
                  "label": int64_feature(labels_list[i]),
                  'img_raw': bytes_feature(img_raw)
              }))  # example对象对label和image数据进行封装
              writer.write(example.SerializeToString())  # 序列化为字符串

    


def read_and_decode(filename_queue):  # up_or_down_train.tfrecords

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来
    img = features['img_raw']
    img = tf.image.decode_png(img, channels=3)  # 这里，也可以解码为 1 通道
    img = tf.image.resize_images(img, [128, 128]) 
    # img = tf.reshape(img, [128, 128, 3])  # 128*128*3
    
    # img_decode = tf.reshape(img_decode, [128, 128, 2])  # reshape为128*128的3通道图片
    # img_decode = tf.cast(img_decode, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img, label


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  num_preprocess_threads = 16
  images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def inputs(eval_data, data_dir, batch_size):
  """Construct input for MACD/KDJ evaluation using the Reader ops.

  Args:
    eval_data: bool, indicating if one should use the train or eval data set.
    data_dir: Path to the MACD/KDJ data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  if not eval_data:
    filenames = [cwd+data_dir]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
  # else:
  #   filenames = [os.path.join(data_dir, 'test_batch.bin')]
  #   num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  print(filenames)
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  image, label = read_and_decode(filename_queue)
  image = tf.cast(image, tf.float32)

  # Subtract off the mean and divide by the variance of the pixels.
  float_image = tf.image.per_image_standardization(image)

  # height = IMAGE_SIZE
  # width = IMAGE_SIZE
  # # Set the shapes of tensors.
  # float_image.set_shape([height, width, 3])
  # label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size)


# inputs(False, "tfrecords\\macd_train.tfrecord", 128)

encode_and_write(stocks_list, "macd_pic\\", "tfrecords\\macd_train.tfrecord")

# with tf.Session() as sess:
#     filename_queue = tf.train.string_input_producer(['F:\Code\\buysell\data\pic_data\\tfrecords\\macd_train.tfrecord'])
#     reshaped_image = read_and_decode(filename_queue)
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
#         img, label = sess.run(reshaped_image)
#         # img = np.fromstring(img, np.uint8)  
#         # img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#         # img = tf.cast(img, tf.float32)
#         # print(img.dtype)
#         # print(sess.run(tf.shape(img)))
#         # 将图片保存
#         # ds = pd.DataFrame(img.tolist())
#         # ds.to_csv('F:\Code\\buysell\data\pic_data\\raw_test\\%d.csv' % i)
#         # print(img)   
#         cv2.imwrite('F:\Code\\buysell\data\pic_data\\raw_test\\%d.png' % i, img)
#         print(label)