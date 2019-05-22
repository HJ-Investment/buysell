
import tensorflow as tf
import numpy as np

def conv(layer_name, x, out_channels, kernel_size=[3,3], stride=[1,1,1,1]):
    '''Convolution op wrapper, use RELU activation after convolution
    Args:
        layer_name: e.g. conv1, pool1...
        x: input tensor, [batch_size, height, width, channels]
        out_channels: number of output channels (or comvolutional kernels)
        kernel_size: the size of convolutional kernel, VGG paper used: [3,3]
        stride: A list of ints. 1-D of length 4. VGG paper used: [1, 1, 1, 1]
        is_pretrain: if load pretrained parameters, freeze all conv layers. 
        Depending on different situations, you can just set part of conv layers to be freezed.
        the parameters of freezed layers will not change when training.
    Returns:
        4D tensor
    '''

    in_channels = x.get_shape()[-1]
    with tf.variable_scope(layer_name):
        w = tf.get_variable(name='weights',
                            trainable=True,
                            shape=[kernel_size[0], kernel_size[1], in_channels, out_channels],
                            initializer=tf.contrib.layers.xavier_initializer()) # default is uniform distribution initialization
        b = tf.get_variable(name='biases',
                            trainable=True,
                            shape=[out_channels],
                            initializer=tf.constant_initializer(0.0))
        x = tf.nn.conv2d(x, w, stride, padding='SAME', name='conv')
        x = tf.nn.bias_add(x, b, name='bias_add')
        x = tf.nn.relu(x, name='relu')
        return x

def pool(layer_name, x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True):
    '''Pooling op
    Args:
        x: input tensor
        kernel: pooling kernel, VGG paper used [1,2,2,1], the size of kernel is 2X2
        stride: stride size, VGG paper used [1,2,2,1]
        padding:
        is_max_pool: boolen
                    if True: use max pooling
                    else: use avg pooling
    '''
    if is_max_pool:
        x = tf.nn.max_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    else:
        x = tf.nn.avg_pool(x, kernel, strides=stride, padding='SAME', name=layer_name)
    return x

def batch_norm(x):
    '''Batch normlization(I didn't include the offset and scale)
    '''
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x, [0])
    x = tf.nn.batch_normalization(x,
                                  mean=batch_mean,
                                  variance=batch_var,
                                  offset=None,
                                  scale=None,
                                  variance_epsilon=epsilon)
    return x

def FC_layer(layer_name, x, out_nodes):
    '''Wrapper for fully connected layers with RELU activation as default
    Args:
        layer_name: e.g. 'FC1', 'FC2'
        x: input feature map
        out_nodes: number of neurons for current FC layer
    '''
    shape = x.get_shape()
    if len(shape) == 4:
        size = shape[1].value * shape[2].value * shape[3].value
    else:
        size = shape[-1].value

    with tf.variable_scope(layer_name):
        w = tf.get_variable('weights',
                            shape=[size, out_nodes],
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable('biases',
                            shape=[out_nodes],
                            initializer=tf.constant_initializer(0.0))
        flat_x = tf.reshape(x, [-1, size]) # flatten into 1D
        
        x = tf.nn.bias_add(tf.matmul(flat_x, w), b)
        x = tf.nn.relu(x,name=layer_name)
        return x


def VGG16N(x, n_classes):
    
    with tf.name_scope('VGG16'):

        x = conv('conv1_1', x, 64, kernel_size=[3,3], stride=[1,1,1,1])   
        x = conv('conv1_2', x, 64, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool1'):    
            x = pool('pool1', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            
        x = conv('conv2_1', x, 128, kernel_size=[3,3], stride=[1,1,1,1])    
        x = conv('conv2_2', x, 128, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool2'):    
            x = pool('pool2', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
         
            

        x = conv('conv3_1', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
        x = conv('conv3_2', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
        x = conv('conv3_3', x, 256, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool3'):
            x = pool('pool3', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
            

        x = conv('conv4_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = conv('conv4_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = conv('conv4_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool4'):
            x = pool('pool4', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)
        

        x = conv('conv5_1', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = conv('conv5_2', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        x = conv('conv5_3', x, 512, kernel_size=[3,3], stride=[1,1,1,1])
        with tf.name_scope('pool5'):
            x = pool('pool5', x, kernel=[1,2,2,1], stride=[1,2,2,1], is_max_pool=True)            
        
        
        x = FC_layer('fc6', x, out_nodes=4096)        
        with tf.name_scope('batch_norm1'):
            x = batch_norm(x)           
        x = FC_layer('fc7', x, out_nodes=4096)        
        with tf.name_scope('batch_norm2'):
            x = batch_norm(x)            
        x = FC_layer('fc8', x, out_nodes=n_classes)
    
        return x

            
