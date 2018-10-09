from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

model_path = 'F:/Code/buysell/slim/macd_j/frozen_graph.pb'
label_path = 'F:/Code/buysell/data/tfrecords/label.txt'
num_top_predictions = 2

class NodeLookup(object):
  def __init__(self, label_lookup_path=None):
    self.node_lookup = self.load(label_lookup_path)

  def load(self, label_lookup_path):
    node_id_to_name = {}
    with open(label_lookup_path) as f:
      for index, line in enumerate(f):
        node_id_to_name[index] = line.strip()
    print(node_id_to_name)
    return node_id_to_name

  def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
      return ''
    return self.node_lookup[node_id]


def create_graph(model_path):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image

def run_inference_on_image(image):
  """Runs inference on an image.
  Args:
    image: Image file name.
  Returns:
    Nothing
  """
  with tf.Graph().as_default():
    image_data = tf.gfile.FastGFile(image, 'rb').read()
    image_data = tf.image.decode_jpeg(image_data)
    image_data = preprocess_for_eval(image_data, 299, 299)
    image_data = tf.expand_dims(image_data, 0)
    with tf.Session() as sess:
      image_data = sess.run(image_data)

  # Creates graph from saved GraphDef.
  # create_graph()

  with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Logits/SpatialSqueeze:0')
    predictions = sess.run(softmax_tensor,
                           {'input:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = NodeLookup(label_path)

    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    up_score = predictions[1]
    notup_score = predictions[0]
    for node_id in top_k:
      human_string = node_lookup.id_to_string(node_id)
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
    if up_score - notup_score > 0.1:
      return 1
    else:
      return 0

