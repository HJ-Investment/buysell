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

slim = tf.contrib.slim
flags = tf.app.flags

flags.DEFINE_string('gpu_indices', '0,1', 'The index of gpus to used.')
flags.DEFINE_string('train_record_path', 
                    'F:/Code/buysell/read_picture/ResNet/datasets/train.record', 
                    'Path to training tfrecord file.')
flags.DEFINE_string('val_record_path', 
                    'F:/Code/buysell/read_picture/ResNet/datasets/val.record', 
                    'Path to validation tfrecord file.')
flags.DEFINE_string('checkpoint_path',
                    'F:/Code/buysell/read_picture/ResNet/datasets/resnet_v1_50.ckpt',
                    'Path to a pretrained model.')
flags.DEFINE_string('model_dir', 'F:/Code/buysell/read_picture/ResNet/training', 'Path to log directory.')
flags.DEFINE_float('keep_checkpoint_every_n_hours', 
                   0.2,
                   'Save model checkpoint every n hours.')
flags.DEFINE_string('learning_rate_decay_type',
                    'exponential',
                    'Specifies how the learning rate is decayed. One of '
                    '"fixed", "exponential", or "polynomial"')
flags.DEFINE_float('learning_rate', 
                   0.0001, 
                   'Initial learning rate.')
flags.DEFINE_float('end_learning_rate', 
                   0.000001,
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
flags.DEFINE_integer('num_classes', 2, 'Number of classes.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_steps', 5000, 'Number of steps.')
flags.DEFINE_integer('input_size', 224, 'Size of input.')
flags.DEFINE_integer('num_gpus', 2, 'Number of gpus')

FLAGS = flags.FLAGS

class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []
    def before_run(self, run_context):
        self.iter_time_start = time.time()
    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)


def get_decoder():
    """Returns a TFExampleDecoder."""
    keys_to_features = {
        'image/encoded': 
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': 
            tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': 
            tf.FixedLenFeature([1], tf.int64, default_value=tf.zeros([1], 
                               dtype=tf.int64))}
        
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                              format_key='image/format',
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('image/class/label', shape=[])}
    
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    return decoder
    
    
def transform_data(image):
    size = FLAGS.input_size + 32
    image = tf.squeeze(tf.image.resize_bilinear([image], size=[size, size]))
    image = tf.to_float(image)
    return image


def read_dataset(file_read_fun, input_files, num_readers=1, shuffle=False,
                 num_epochs=0, read_block_length=32, shuffle_buffer_size=2048):
    """Reads a dataset, and handles repeatition and shuffling.
    
    This function and the following are modified from:
        https://github.com/tensorflow/models/blob/master/research/
            object_detection/builders/dataset_builder.py
    
    Args:
        file_read_fun: Function to use in tf.contrib.data.parallel_iterleave,
            to read every individual file into a tf.data.Dataset.
        input_files: A list of file paths to read.
        
    Returns:
        A tf.data.Dataset of (undecoded) tf-records.
    """
    # Shard, shuffle, and read files
    filenames = tf.gfile.Glob(input_files)
    if num_readers > len(filenames):
        num_readers = len(filenames)
        tf.logging.warning('num_readers has been reduced to %d to match input '
                           'file shards.' % num_readers)
    filename_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle:
        filename_dataset = filename_dataset.shuffle(100)
    elif num_readers > 1:
        tf.logging.warning('`shuffle` is false, but the input data stream is '
                           'still slightly shuffled since `num_readers` > 1.')
    filename_dataset = filename_dataset.repeat(num_epochs or None)
    records_dataset = filename_dataset.apply(
        tf.contrib.data.parallel_interleave(
            file_read_fun,
            cycle_length=num_readers,
            block_length=read_block_length,
            sloppy=shuffle))
    if shuffle:
        records_dataset = records_dataset.shuffle(shuffle_buffer_size)
    return records_dataset  


def create_input_fn(record_paths, batch_size=64,
                    num_epochs=0, num_parallel_batches=8, 
                    num_prefetch_batches=2):
    """Create a train or eval `input` function for `Estimator`.
    
    Args:
        record_paths: A list contains the paths of tfrecords.
    
    Returns:
        `input_fn` for `Estimator` in TRAIN/EVAL mode.
    """
    def _input_fn():
        decoder = get_decoder()
        
        def decode(value):
            keys = decoder.list_items()
            tensors = decoder.decode(value)
            tensor_dict = dict(zip(keys, tensors))
            image = tensor_dict.get('image')
            image = transform_data(image)
            features_dict = {'image': image}
            return features_dict, tensor_dict.get('label')
        
        dataset = read_dataset(
            functools.partial(tf.data.TFRecordDataset, 
                              buffer_size=8 * 1000 * 1000),
            input_files=record_paths,
            num_epochs=num_epochs)
        if batch_size:
            num_parallel_calles = batch_size * num_parallel_batches
        else:
            num_parallel_calles = num_parallel_batches
        dataset = dataset.map(decode, num_parallel_calls=num_parallel_calles)
        if batch_size:
            dataset = dataset.apply(
                tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(num_prefetch_batches)
        return dataset
    
    return _input_fn


def create_predict_input_fn():
    """Creates a predict `input` function for `Estimator`.
    
    Modified from:
        https://github.com/tensorflow/models/blob/master/research/
            object_detection/inputs.py
    
    Returns:
        `input_fn` for `Estimator` in PREDICT mode.
    """
    def _predict_input_fn():
        """Decodes serialized tf.Examples and returns `ServingInputReceiver`.
        
        Returns:
            `ServingInputReceiver`.
        """
        example = tf.placeholder(dtype=tf.string, shape=[], name='tf_example')
        
        decoder = get_decoder()
        keys = decoder.list_items()
        tensors = decoder.decode(example, items=keys)
        tensor_dict = dict(zip(keys, tensors))
        image = tensor_dict.get('image')
        image = transform_data(image)
        images = tf.expand_dims(image, axis=0)
        return tf.estimator.export.ServingInputReceiver(
            features={'image': images},
            receiver_tensors={'serialized_example': example})
        
    return _predict_input_fn


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
    
    cls_model = model.Model(is_training=is_training, 
                            num_classes=FLAGS.num_classes)
    preprocessed_inputs = cls_model.preprocess(features.get('image'))
    prediction_dict = cls_model.predict(preprocessed_inputs)
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


def resnet_model_fn(features, labels, mode, model_class,
                    resnet_size, weight_decay, learning_rate_fn, momentum,
                    data_format, resnet_version, loss_scale,
                    loss_filter_fn=None, dtype=resnet_model.DEFAULT_DTYPE,
                    fine_tune=False, label_smoothing=0.0):
  """Shared functionality for different resnet model_fns.
  Initializes the ResnetModel representing the model layers
  and uses that model to build the necessary EstimatorSpecs for
  the `mode` in question. For training, this means building losses,
  the optimizer, and the train op that get passed into the EstimatorSpec.
  For evaluation and prediction, the EstimatorSpec is returned without
  a train op, but with the necessary parameters for the given mode.
  Args:
    features: tensor representing input images
    labels: tensor representing class labels for all input images
    mode: current estimator mode; should be one of
      `tf.estimator.ModeKeys.TRAIN`, `EVALUATE`, `PREDICT`
    model_class: a class representing a TensorFlow model that has a __call__
      function. We assume here that this is a subclass of ResnetModel.
    resnet_size: A single integer for the size of the ResNet model.
    weight_decay: weight decay loss rate used to regularize learned variables.
    learning_rate_fn: function that returns the current learning rate given
      the current global_step
    momentum: momentum term used for optimization
    data_format: Input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
    resnet_version: Integer representing which version of the ResNet network to
      use. See README for details. Valid values: [1, 2]
    loss_scale: The factor to scale the loss for numerical stability. A detailed
      summary is present in the arg parser help text.
    loss_filter_fn: function that takes a string variable name and returns
      True if the var should be included in loss calculation, and False
      otherwise. If None, batch_normalization variables will be excluded
      from the loss.
    dtype: the TensorFlow dtype to use for calculations.
    fine_tune: If True only train the dense layers(final layers).
    label_smoothing: If greater than 0 then smooth the labels.
  Returns:
    EstimatorSpec parameterized according to the input params and the
    current mode.
  """

  # Generate a summary node for the images
  tf.compat.v1.summary.image('images', features, max_outputs=6)
  # Checks that features/images have same data type being used for calculations.
  assert features.dtype == dtype

  model = model_class(resnet_size, data_format, resnet_version=resnet_version,
                      dtype=dtype)

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
  if label_smoothing != 0.0:
    one_hot_labels = tf.one_hot(labels, 1001)
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=one_hot_labels,
        label_smoothing=label_smoothing)
  else:
    cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)

  # Create a tensor named cross_entropy for logging purposes.
  tf.identity(cross_entropy, name='cross_entropy')
  tf.compat.v1.summary.scalar('cross_entropy', cross_entropy)

  # If no loss_filter_fn is passed, assume we want the default behavior,
  # which is that batch_normalization variables are excluded from loss.
  def exclude_batch_norm(name):
    return 'batch_normalization' not in name
  loss_filter_fn = loss_filter_fn or exclude_batch_norm

  # Add weight decay to the loss.
  l2_loss = weight_decay * tf.add_n(
      # loss is computed using fp32 for numerical stability.
      [
          tf.nn.l2_loss(tf.cast(v, tf.float32))
          for v in tf.compat.v1.trainable_variables()
          if loss_filter_fn(v.name)
      ])
  tf.compat.v1.summary.scalar('l2_loss', l2_loss)
  loss = cross_entropy + l2_loss

  if mode == tf.estimator.ModeKeys.TRAIN:
    global_step = tf.compat.v1.train.get_or_create_global_step()

    learning_rate = learning_rate_fn(global_step)

    # Create a tensor named learning_rate for logging purposes
    tf.identity(learning_rate, name='learning_rate')
    tf.compat.v1.summary.scalar('learning_rate', learning_rate)

    if flags.FLAGS.enable_lars:
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

    fp16_implementation = getattr(flags.FLAGS, 'fp16_implementation', None)
    if fp16_implementation == 'graph_rewrite':
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer, loss_scale=loss_scale)

    def _dense_grad_filter(gvs):
      """Only apply gradient updates to the final layer.
      This function is used for fine tuning.
      Args:
        gvs: list of tuples with gradients and variable info
      Returns:
        filtered gradients so that only the dense layer remains
      """
      return [(g, v) for g, v in gvs if 'dense' in v.name]

    if loss_scale != 1 and fp16_implementation != 'graph_rewrite':
      # When computing fp16 gradients, often intermediate tensor values are
      # so small, they underflow to 0. To avoid this, we multiply the loss by
      # loss_scale to make these tensor values loss_scale times bigger.
      scaled_grad_vars = optimizer.compute_gradients(loss * loss_scale)

      if fine_tune:
        scaled_grad_vars = _dense_grad_filter(scaled_grad_vars)

      # Once the gradient computation is complete we can scale the gradients
      # back to the correct scale before passing them to the optimizer.
      unscaled_grad_vars = [(grad / loss_scale, var)
                            for grad, var in scaled_grad_vars]
      minimize_op = optimizer.apply_gradients(unscaled_grad_vars, global_step)
    else:
      grad_vars = optimizer.compute_gradients(loss)
      if fine_tune:
        grad_vars = _dense_grad_filter(grad_vars)
      minimize_op = optimizer.apply_gradients(grad_vars, global_step)

    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
    train_op = tf.group(minimize_op, update_ops)
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
      eval_metric_ops=metrics)

    
    
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_indices

    # strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=FLAGS.num_gpus)
    strategy = tf.distribute.MirroredStrategy()
    # session_config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    config = tf.estimator.RunConfig(train_distribute=strategy,
                                    save_checkpoints_secs=120)

    # tf.estimator.Estimator(model_fn, model_dir=None, config=None, params=None, warm_start_from=None)
    # model_fn 是模型函数；
    # model_dir 是训练时模型保存的路径；
    # config 是 tf.estimator.RunConfig 的配置对象；
    # params 是传入 model_fn 的超参数字典；
    # warm_start_from 或者是一个预训练文件的路径，或者是一个 tf.estimator.WarmStartSettings 对象，用于完整的配置热启动参数。
    estimator = tf.estimator.Estimator(model_fn=create_model_fn, 
                                       model_dir=FLAGS.model_dir,
                                       config=config)

    classifier = tf.estimator.Estimator(model_fn=model_function,
                                        model_dir=flags_obj.model_dir,
                                        config=run_config,
                                        warm_start_from=warm_start_settings,
                                        params={
                                                    'resnet_size': int(flags_obj.resnet_size),
                                                    'data_format': flags_obj.data_format,
                                                    'batch_size': flags_obj.batch_size,
                                                    'resnet_version': int(flags_obj.resnet_version),
                                                    'loss_scale': flags_core.get_loss_scale(flags_obj,
                                                                                            default_for_fp16=128),
                                                    'dtype': flags_core.get_tf_dtype(flags_obj),
                                                    'fine_tune': flags_obj.fine_tune,
                                                    'num_workers': num_workers,
                                                })


    train_input_fn = create_input_fn([FLAGS.train_record_path], 
                                     batch_size=FLAGS.batch_size)

    # 使用 tf.estimator.TrainSpec 指定训练输入函数及相关参数。该类的完整形式是：
    # tf.estimator.TrainSpec(input_fn, max_steps, hooks)
    # input_fn 用来提供训练时的输入数据；
    # max_steps 指定总共训练多少步；
    # hooks 是一个 tf.train.SessionRunHook 对象，用来配置分布式训练等参数。
    time_hist = TimeHistory()
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=FLAGS.num_steps,
                                        hooks=[time_hist])

    eval_input_fn = create_input_fn([FLAGS.val_record_path], 
                                    batch_size=FLAGS.batch_size,
                                    num_epochs=1)

    predict_input_fn = create_predict_input_fn()
    
    eval_exporter = tf.estimator.FinalExporter(
        name='servo', serving_input_receiver_fn=predict_input_fn)

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
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=None,
                                      exporters=eval_exporter)
    
    # estimator 是一个 tf.estimator.Estimator 对象，用于指定模型函数以及其它相关参数；
    # train_spec 是一个 tf.estimator.TrainSpec 对象，用于指定训练的输入函数以及其它参数；
    # eval_spec 是一个 tf.estimator.EvalSpec 对象，用于指定验证的输入函数以及其它参数。
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    total_time = sum(time_hist.times)
    print(f"total time with {FLAGS.num_gpus} GPU(s): {total_time} seconds")
    avg_time_per_batch = np.mean(time_hist.times)
    print(f"{FLAGS.batch_size*FLAGS.num_gpus/avg_time_per_batch} images/second with {FLAGS.num_gpus} GPU(s)")

    
if __name__ == '__main__':
    tf.app.run()