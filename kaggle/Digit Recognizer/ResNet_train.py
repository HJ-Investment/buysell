#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
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
import tensorflow as tf

import ResNet
import input_data
import config
import exporter

import pandas as pd

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('gpu_indices', '0', 'The index of gpus to used.')

flags.DEFINE_string('train_record_path', 
                    config.train_record_path,
                    'Path to training tfrecord file.')

flags.DEFINE_string('val_record_path', 
                    config.val_record_path,
                    'Path to validation tfrecord file.')

flags.DEFINE_string('checkpoint_path',
                    None,
                    'Path to a pretrained model.')

flags.DEFINE_string('model_dir',
                    config.model_dir,
                    'Path to log directory.')

flags.DEFINE_float('keep_checkpoint_every_n_hours', 
                   0.05,
                   'Save model checkpoint every n hours.')

flags.DEFINE_string('learning_rate_decay_type',
                    'exponential',
                    'Specifies how the learning rate is decayed. One of '
                    '"fixed", "exponential", or "polynomial"')

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

flags.DEFINE_integer('num_classes', 10, 'Number of classes.')

flags.DEFINE_integer('batch_size', 64, 'Batch size.')

flags.DEFINE_integer('num_steps', 5000, 'Number of steps.')

flags.DEFINE_integer('input_size', 224, 'Number of steps.')

FLAGS = flags.FLAGS


def create_input_fn(record_paths, batch_size=64):
    """Create a train or eval `input` function for `Estimator`.
    
    Args:
        record_paths: A list contains the paths of train.csv.
    
    Returns:
        `input_fn` for `Estimator` in TRAIN/EVAL mode.
    """
    def _input_fn():
        df = pd.read_csv(record_paths, sep=',')
        dataset = input_data.get_resnet(df, batch_size)
        return dataset
    
    return _input_fn


def create_model_fn(features, labels, mode, params=None):
    """Constructs the classification model.
    
    Modifed from:
        https://github.com/tensorflow/models/blob/master/research/
            object_detection/model_lib.py.
    
    Args:
        features: A 4-D float32 tensor with shape [batch_size, height,
            width, channels] representing a batch of images. (Support dict) 可以是一个张量，也可以是由张量组成的一个字典；
        labels: A 1-D int32 tensor with shape [batch_size] representing
             the labels of each image. (Support dict) 可以是一个张量，也可以是由张量组成的一个字典；
        mode: Mode key for tf.estimator.ModeKeys. 指定训练模式，可以取 （TRAIN, EVAL, PREDICT）三者之一；
        params: Parameter dictionary passed from the estimator. 是一个（可要可不要的）字典，指定其它超参数。 
        
    Returns:
        An `EstimatorSpec` the encapsulates the model and its serving
        configurations.
    """
    params = params or {}
    loss, acc, train_op, export_outputs = None, None, None, None
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    
    cls_model = ResNet.Model(is_training=is_training,
                            num_classes=FLAGS.num_classes)
    # preprocessed_inputs = cls_model.preprocess(features.get('image'))
    prediction_dict = cls_model.predict(features)
    postprocessed_dict = cls_model.postprocess(prediction_dict)
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.checkpoint_path:
            init_variables_from_checkpoint()
    
    if mode in (tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL):
        loss_dict = cls_model.loss(prediction_dict, labels)
        loss = loss_dict['loss']
        classes = postprocessed_dict['classes']
        acc = tf.reduce_mean(tf.cast(tf.equal(classes, labels), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', acc)
    
    scaffold = None
    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()
        learning_rate = configure_learning_rate(FLAGS.decay_steps,
                                                global_step)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                               momentum=0.9)
        train_op = slim.learning.create_train_op(loss, optimizer,
                                                 summarize_gradients=True)
        
        keep_checkpoint_every_n_hours = FLAGS.keep_checkpoint_every_n_hours
        saver = tf.train.Saver(
            sharded=True,
            keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        scaffold = tf.train.Scaffold(saver=saver)
        
    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(labels=labels, predictions=classes)
        eval_metric_ops = {'Accuracy': accuracy}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        export_output = exporter._add_output_tensor_nodes(postprocessed_dict)
        export_outputs = {
            tf.saved_model.signature_constants.PREDICT_METHOD_NAME:
                tf.estimator.export.PredictOutput(export_output)}

    # tf.estimator.EstimatorSpec(
        # mode, 指定当前是处于训练、验证还是预测状态
        # predictions=None, 预测的一个张量，或者是由张量组成的一个字典
        # loss=None, 损失张量
        # train_op=None, 指定优化操作
        # eval_metric_ops=None, 指定各种评估度量的字典，这个字典的值必须是如下两种形式： Metric 类的实例； 调用某个评估度量函数的结果对 (metric_tensor, update_op)；
        # export_outputs=None, 用于模型保存，描述了导出到 SavedModel 的输出格式
        # training_chief_hooks=None,
        # training_hooks=None,
        # scaffold=None, 一个 tf.train.Scaffold 对象，可以在训练阶段初始化、保存等时使用
        # evaluation_hooks=None,
        # prediction_hooks=None)    
    return tf.estimator.EstimatorSpec(mode=mode,
                                      predictions=prediction_dict,
                                      loss=loss,
                                      train_op=train_op,
                                      eval_metric_ops=eval_metric_ops,
                                      export_outputs=export_outputs,
                                      scaffold=scaffold)
    
    
def configure_learning_rate(decay_steps, global_step):
    """Configures the learning rate.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/slim/
        train_image_classifier.py
    
    Args:
        decay_steps: The step to decay learning rate.
        global_step: The global_step tensor.
        
    Returns:
        A `Tensor` representing the learning rate.
    """ 
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
        
        
def init_variables_from_checkpoint(checkpoint_exclude_scopes=None):
    """Variable initialization form a given checkpoint path.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/
        object_detection/model_lib.py
    
    Note that the init_fn is only run when initializing the model during the 
    very first global step.
    
    Args:
        checkpoint_exclude_scopes: Comma-separated list of scopes of variables
            to exclude when restoring from a checkpoint.
    """
    exclude_patterns = None
    if checkpoint_exclude_scopes:
        exclude_patterns = [scope.strip() for scope in 
                            checkpoint_exclude_scopes.split(',')]
    variables_to_restore = tf.global_variables()
    variables_to_restore.append(slim.get_or_create_global_step())
    variables_to_init = tf.contrib.framework.filter_variables(
        variables_to_restore, exclude_patterns=exclude_patterns)
    variables_to_init_dict = {var.op.name: var for var in variables_to_init}
    
    available_var_map = get_variables_available_in_checkpoint(
        variables_to_init_dict, FLAGS.checkpoint_path, 
        include_global_step=False)
    tf.train.init_from_checkpoint(FLAGS.checkpoint_path, available_var_map)
    
    
def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
    """Returns the subset of variables in the checkpoint.
    
    Inspects given checkpoint and returns the subset of variables that are
    available in it.
    
    Args:
        variables: A dictionary of variables to find in checkpoint.
        checkpoint_path: Path to the checkpoint to restore variables from.
        include_global_step: Whether to include `global_step` variable, if it
            exists. Default True.
            
    Returns:
        A dictionary of variables.
        
    Raises:
        ValueError: If `variables` is not a dict.
    """
    if not isinstance(variables, dict):
        raise ValueError('`variables` is expected to be a dict.')
    
    # Available variables
    ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
    ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
    if not include_global_step:
        ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
    vars_in_ckpt = {}
    for variable_name, variable in sorted(variables.items()):
        if variable_name in ckpt_vars_to_shape_map:
            if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
                vars_in_ckpt[variable_name] = variable
            else:
                logging.warning('Variable [%s] is avaible in checkpoint, but '
                                'has an incompatible shape with model '
                                'variable. Checkpoint shape: [%s], model '
                                'variable shape: [%s]. This variable will not '
                                'be initialized from the checkpoint.',
                                variable_name, 
                                ckpt_vars_to_shape_map[variable_name],
                                variable.shape.as_list())
        else:
            logging.warning('Variable [%s] is not available in checkpoint',
                            variable_name)
    return vars_in_ckpt


def main(_):
    # Specify which gpu to be used
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_indices

    session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    config = tf.estimator.RunConfig(session_config=session_config)
    # tf.estimator.Estimator(model_fn, model_dir=None, config=None, params=None, warm_start_from=None)
    # model_fn 是模型函数；
    # model_dir 是训练时模型保存的路径；
    # config 是 tf.estimator.RunConfig 的配置对象；
    # params 是传入 model_fn 的超参数字典；
    # warm_start_from 或者是一个预训练文件的路径，或者是一个 tf.estimator.WarmStartSettings 对象，用于完整的配置热启动参数。
    estimator = tf.estimator.Estimator(model_fn=create_model_fn, 
                                       model_dir=FLAGS.model_dir,
                                       config=config)

    train_input_fn = create_input_fn(FLAGS.train_record_path,
                                     batch_size=FLAGS.batch_size)

    # 使用 tf.estimator.TrainSpec 指定训练输入函数及相关参数。该类的完整形式是：
    # tf.estimator.TrainSpec(input_fn, max_steps, hooks)
    # input_fn 用来提供训练时的输入数据；max_steps 指定总共训练多少步；hooks 是一个 tf.train.SessionRunHook 对象，用来配置分布式训练等参数。
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=FLAGS.num_steps)

    eval_input_fn = create_input_fn(FLAGS.val_record_path,
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