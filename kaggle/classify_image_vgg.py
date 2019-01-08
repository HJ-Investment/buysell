from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
import tensorflow as tf

import input_data

FLAGS = None


def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(FLAGS.model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def run_inference_on_image():
  df = pd.read_csv('./kaggle/test.csv', sep=',')
  test_batch = input_data.val(df)

  # Creates graph from saved GraphDef.
  create_graph()

  for step in np.arange(len(df.index)):   
    with tf.Session() as sess:
        test_images = sess.run(test_batch)

        softmax_tensor = sess.graph.get_tensor_by_name('VGG16/fc8/fc8:0')
        predictions = sess.run(softmax_tensor,
                            {'fc8:0': test_images})
        predictions = np.squeeze(predictions)

        top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
        score = predictions[top_k[0]]
        print('%s (score = %.5f)' % (top_k[0], score))


def restore_model_ckpt():
    ckpt_file_path = 'F:\\Code\\buysell\\kaggle\\output\\train\\'
    saver = tf.train.import_meta_graph(ckpt_file_path + 'model.ckpt-1999.meta')  # 加载模型结构
    df = pd.read_csv('./kaggle/test.csv', sep=',')
    test_batch = input_data.val(df, 32)

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(ckpt_file_path))  # 只需要指定目录就可以恢复所有变量信息
        graph = tf.get_default_graph()
        x = graph.get_tensor_by_name("x:0")
        softmax_tensor = graph.get_tensor_by_name('logits_eval:0')
        all_predictions = []
        for step in np.arange(len(df.index)/32):
            test_images = sess.run(test_batch)

            predictions = sess.run(softmax_tensor,
                              {x: test_images})
            #打印出预测矩阵
            # print(predictions)
            #打印出预测矩阵每一行最大值的索引
            value = tf.argmax(predictions,1).eval()
            print(value)
            all_predictions.append(value.tolist())
            del value
        datas = pd.DataFrame(all_predictions)
        print(datas)

def main(_):
  restore_model_ckpt()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--model_path',
      default='F:\\Code\\buysell\\kaggle\\output\\train\\vgg_model_frozen.pb',
      type=str,
  )
  parser.add_argument(
      '--num_top_predictions',
      type=int,
      default=1,
      help='Display this many predictions.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

