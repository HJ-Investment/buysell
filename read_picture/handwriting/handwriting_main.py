#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 19:27:44 2018

@author: shirhe-lyh


Train a CNN model to classifying 10 digits.

Example Usage:
---------------
python3 train.py \
    --train_record_path: Path to training tfrecord file.
    --val_record_path: Path to validation tfrecord file.
    --model_dir: Path to log directory.
"""

import functools
import logging
import os
import time
import tensorflow as tf

import exporter
import resnet_model

from official.utils.logs import hooks_helper
from official.utils.misc import distribution_utils

import input_data

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('gpu_indices', '0,1', 'The index of gpus to used.')
flags.DEFINE_string('train_record_path',
                    '/tf/handwriting/datasets/',
                    'Path to training tfrecord file.')
flags.DEFINE_string('val_record_path',
                    '/tf/handwriting/datasets/',
                    'Path to validation tfrecord file.')
flags.DEFINE_string('checkpoint_path',
                    '/tf/handwriting/datasets/resnet/training',
                    'Path to a pretrained model.')
flags.DEFINE_string('model_dir', '/tf/handwriting/training', 'Path to log directory.')
# flags.DEFINE_float('keep_checkpoint_every_n_hours',
#                    0.2,
#                    'Save model checkpoint every n hours.')
flags.DEFINE_string('learning_rate_decay_type',
                    'exponential',
                    'Specifies how the learning rate is decayed. One of '
                    '"fixed", "exponential", or "polynomial"')
flags.DEFINE_integer('pretrained_model_checkpoint_path', None, '')
flags.DEFINE_float('learning_rate',
                   0.01,
                   'Initial learning rate.')
flags.DEFINE_float('end_learning_rate',
                   0.0001,
                   'The minimal end learning rate used by a polynomial decay '
                   'learning rate.')
flags.DEFINE_float('decay_steps',
                   1000,
                   'Number of epochs after which learning rate decays. '
                   'Note: this flag counts epochs per clone but aggregates '
                   'per sync replicas. So 1.0 means that each clone will go '
                   'over full epoch individually, but replicas will go once '
                   'across all replicas.')
flags.DEFINE_float('learning_rate_decay_factor',
                   0.5,
                   'Learning rate decay factor.')
flags.DEFINE_float('label_smoothing',
                   0,
                   'Label smoothing.')
flags.DEFINE_float('momentum',
                   0.9,
                   'Momentum.')
flags.DEFINE_float('weight_decay',
                   1e-4,
                   'Weight decay coefficiant for l2 regularization.')
flags.DEFINE_integer('num_classes', 26, 'Number of classes.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_steps', 50000, 'Number of steps.')
flags.DEFINE_integer('input_size', 224, 'Size of input.')
flags.DEFINE_integer('num_gpus', 2, 'Number of gpus')
flags.DEFINE_boolean('loss_scale', 1, "Loss scale")
flags.DEFINE_boolean('fp16_implementation', 1, 'fp16 implementation')
flags.DEFINE_boolean('enable_lars', False, '')

FLAGS = flags.FLAGS


def transform_data(image):
    size = FLAGS.input_size
    image = tf.squeeze(tf.image.resize_bilinear([image], size=[size, size]))
    image = tf.to_float(image)
    return image


def create_input_fn(record_paths, batch_size=64, is_train=True,
                    num_prefetch_batches=2):
    """Create a train or eval `input` function for `Estimator`.

    Args:
        record_paths: A list contains the paths of tfrecords.

    Returns:
        `input_fn` for `Estimator` in TRAIN/EVAL mode.
    """

    def _input_fn():
        if(is_train):
            images = 'emnist-letters-train-images-idx3-ubyte'
            labels = 'emnist-letters-train-labels-idx1-ubyte'
        else:
            images = 'emnist-letters-test-images-idx3-ubyte'
            labels = 'emnist-letters-test-labels-idx1-ubyte'

        dataset = input_data.load_emnist_image(record_paths, images, labels)

        # if batch_size:
        #     dataset = dataset.apply(
        #         tf.contrib.data.batch_and_drop_remainder(batch_size))
        # dataset = dataset.prefetch(num_prefetch_batches)
        return dataset

    return _input_fn


def _get_block_sizes(resnet_size):
    choices = {
        18: [2, 2, 2, 2],
        34: [3, 4, 6, 3],
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3],
        200: [3, 24, 36, 3]
    }
    try:
        return choices[resnet_size]
    except KeyError:
        err = ('Could not find layers for selected Resnet size.\n'
            'Size received: {}; sizes allowed: {}.'.format(
            resnet_size, choices.keys()))
        raise ValueError(err)


def resnet_model_fn(features, labels, mode, params):

    # Generate a summary node for the images
    tf.compat.v1.summary.image('images', features, max_outputs=6)
    # Checks that features/images have same data type being used for calculations.
    assert features.dtype == resnet_model.DEFAULT_DTYPE

    resnet_size = params['resnet_size']
    if resnet_size < 50:
        bottleneck = False
    else:
        bottleneck = True
    model = resnet_model.Model(resnet_size=resnet_size,
                               bottleneck=bottleneck,
                               num_classes=FLAGS.num_classes,
                               num_filters=64,
                               kernel_size=7,
                               conv_stride=2,
                               first_pool_size=3,
                               first_pool_stride=2,
                               block_sizes=_get_block_sizes(resnet_size),
                               block_strides=[1, 2, 2, 2],
                               resnet_version=resnet_model.DEFAULT_VERSION,
                               data_format='channels_first',
                               dtype=resnet_model.DEFAULT_DTYPE)

    logits = model(features, mode == tf.estimator.ModeKeys.TRAIN)

    # This acts as a no-op if the logits are already in fp32 (provided logits are
    # not a SparseTensor). If dtype is is low precision, logits must be cast to
    # fp32 for numerical stability.
    logits = tf.cast(logits, tf.float32)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Return the predictions and the specification for serving a SavedModel
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    if FLAGS.label_smoothing != 0.0:
        one_hot_labels = tf.one_hot(labels, 1001)
        cross_entropy = tf.losses.softmax_cross_entropy(
            logits=logits, onehot_labels=one_hot_labels,
            label_smoothing=FLAGS.label_smoothing)
    else:
        cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
            logits=logits, labels=labels)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

    # If no loss_filter_fn is passed, assume we want the default behavior,
    # which is that batch_normalization variables are excluded from loss.
    def loss_filter_fn(_):
        return True

    # Add weight decay to the loss.
    weight_decay = FLAGS.weight_decay
    l2_loss = weight_decay * tf.add_n(
        # loss is computed using fp32 for numerical stability.
        [
            tf.nn.l2_loss(tf.cast(v, tf.float32))
            for v in tf.compat.v1.trainable_variables()
            if loss_filter_fn(v.name)
            ])
    tf.compat.v1.summary.scalar('l2_loss', l2_loss)
    loss = cross_entropy + l2_loss

    scaffold = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        global_step = tf.compat.v1.train.get_or_create_global_step()
        learning_rate = configure_learning_rate(FLAGS.decay_steps,
                                                global_step)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.compat.v1.summary.scalar('learning_rate', learning_rate)

        momentum = FLAGS.momentum
        if FLAGS.enable_lars:
            optimizer = tf.contrib.opt.LARSOptimizer(
                learning_rate,
                momentum=momentum,
                weight_decay=weight_decay,
                skip_list=['batch_normalization', 'bias'])
        else:
            optimizer = tf.compat.v1.train.MomentumOptimizer(
                learning_rate=learning_rate,
                momentum=momentum
            )

        # loss_scale = FLAGS.loss_scale
        # fp16_implementation = FLAGS.fp16_implementation
        # if fp16_implementation == 'graph_rewrite':
        #     optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
        #         optimizer, loss_scale=loss_scale)

        def _dense_grad_filter(gvs):
            """Only apply gradient updates to the final layer.
            This function is used for fine tuning.
            Args:
              gvs: list of tuples with gradients and variable info
            Returns:
              filtered gradients so that only the dense layer remains
            """
            return [(g, v) for g, v in gvs if 'dense' in v.name]

        # if loss_scale != 1 and fp16_implementation != 'graph_rewrite':
        #     # When computing fp16 gradients, often intermediate tensor values are
        #     # so small, they underflow to 0. To avoid this, we multiply the loss by
        #     # loss_scale to make these tensor values loss_scale times bigger.
        #     scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)
        #
        #     if fine_tune:
        #         scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)
        #
        #     # Once the gradient computation is complete we can scale the gradients
        #     # back to the correct scale before passing them to the optimizer.
        #     unscaled_grad_vars = [(grad / loss_scale, var)
        #                           for grad, var in scaled_grad_vars]
        #     minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
        # else:
        grad_vars = optimizer.compute_gradients(loss)
        # if fine_tune:
        #     grad_vars = _dense_grad_filter(grad_vars)
        minimize_op = optimizer.apply_gradients(grad_vars, global_step)

        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        train_op = tf.group(minimize_op, update_ops)
        

        # keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        # saver = tf.train.Saver(
        #     sharded=True,
        #     keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        #     save_relative_paths=True)
        # # if not tf.get_collection(tf.GraphKeys.SAVERS):
        # tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        # scaffold = tf.train.Scaffold(saver=saver)

    else:
        train_op = None

    accuracy = tf.compat.v1.metrics.accuracy(labels, predictions['classes'])
    accuracy_top_5 = tf.compat.v1.metrics.mean(
        tf.nn.in_top_k(predictions=logits, targets=labels, k=5, name='top_5_op'))
    metrics = {'accuracy': accuracy,
               'accuracy_top_5': accuracy_top_5}

    # Create a tensor named train_accuracy for logging purposes
    tf.identity(accuracy[1], name='train_accuracy')
    tf.identity(accuracy_top_5[1], name='train_accuracy_top_5')
    tf.compat.v1.summary.scalar('train_accuracy', accuracy[1])
    tf.compat.v1.summary.scalar('train_accuracy_top_5', accuracy_top_5[1])

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=metrics,
        scaffold=scaffold)


def configure_learning_rate(decay_steps, global_step):

    if FLAGS.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(FLAGS.learning_rate,
                                          global_step,
                                          decay_steps,
                                          FLAGS.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'fixed':
        return tf.constant(FLAGS.learning_rate, name='fixed_learning_rate')
    elif FLAGS.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(FLAGS.learning_rate,
                                         global_step,
                                         decay_steps,
                                         FLAGS.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized' %
                         FLAGS.learning_rate_decay_type)
        



def main(_):
    # Specify which gpu to be used
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_indices

    # strategy = None
    # According to the guide, https://www.tensorflow.org/guide/distribute_strategy, MirroredStrategy defaults to using NCCL for cross device communication and NCCL is not available on Windows.
    # following code only work at TensorFlow 1.13
    strategy = tf.contrib.distribute.MirroredStrategy()
    # session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7))
    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    save_checkpoints_secs=120)#,
                                    # session_config=session_config)

    if FLAGS.pretrained_model_checkpoint_path is not None:
        warm_start_settings = tf.estimator.WarmStartSettings(
            FLAGS.pretrained_model_checkpoint_path,
            vars_to_warm_start='^(?!.*dense)')
    else:
        warm_start_settings = None
    # tf.estimator.Estimator(model_fn, model_dir=None, config=None, params=None, warm_start_from=None)
    # model_fn 是模型函数；
    # model_dir 是训练时模型保存的路径；
    # config 是 tf.estimator.RunConfig 的配置对象；
    # params 是传入 model_fn 的超参数字典；
    # warm_start_from 或者是一个预训练文件的路径，或者是一个 tf.estimator.WarmStartSettings 对象，用于完整的配置热启动参数。
    estimator = tf.estimator.Estimator(model_fn=resnet_model_fn,
                                       model_dir=FLAGS.model_dir,
                                       config=config,
                                       warm_start_from=warm_start_settings,
                                       params={'resnet_size': 50})

    train_input_fn = create_input_fn(FLAGS.train_record_path, is_train=True,
                                     batch_size=FLAGS.batch_size)

    # 使用 tf.estimator.TrainSpec 指定训练输入函数及相关参数。该类的完整形式是：
    # tf.estimator.TrainSpec(input_fn, max_steps, hooks)
    # input_fn 用来提供训练时的输入数据；
    # max_steps 指定总共训练多少步；
    # hooks 是一个 tf.train.SessionRunHook 对象，用来配置分布式训练等参数。
    # train_hooks = hooks_helper.get_train_hooks(["profilerhook"], 
    #                                            model_dir=FLAGS.model_dir,
    #                                            batch_size=FLAGS.batch_size,
    #                                            save_steps=50,
    #                                            )
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        # hooks=train_hooks,
                                        max_steps=FLAGS.num_steps)

    eval_input_fn = create_input_fn(FLAGS.val_record_path, is_train=False,
                                    batch_size=FLAGS.batch_size)

    # 使用 tf.estimator.EvalSpec 指定验证输入函数及相关参数。该类的完整形式是：
    # tf.estimator.EvalSpec(
    #     input_fn,
    #     steps=100,
    #     name=None,
    #     hooks=None,
    #     exporters=None,
    #     start_delay_secs=120,
    #     throttle_secs=600)
    # 其中 input_fn 用来提供验证时的输入数据；
    # steps 指定总共验证多少步（一般设定为 None 即可）；
    # hooks 用来配置分布式训练等参数；
    # exporters 是一个 Exporter 迭代器，会参与到每次的模型验证；
    # start_delay_secs 指定多少秒之后开始模型验证；
    # throttle_secs 指定多少秒之后重新开始新一轮模型验证（当然，如果没有新的模型断点保存，则该数值秒之后不会进行模型验证，因此这是新一轮模型验证需要等待的最小秒数）
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None)

    # estimator 是一个 tf.estimator.Estimator 对象，用于指定模型函数以及其它相关参数；
    # train_spec 是一个 tf.estimator.TrainSpec 对象，用于指定训练的输入函数以及其它参数；
    # eval_spec 是一个 tf.estimator.EvalSpec 对象，用于指定验证的输入函数以及其它参数。
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    tf.app.run()