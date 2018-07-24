import os
import os.path

import numpy as np
import tensorflow as tf

import decode_tfrecord
import VGG

IMG_W = 256
IMG_H = 256
N_CLASSES = 3
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 10000   # it took me about one hour to complete the training.

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', './data/tfrecords',
                           """Path to the MACD data directory.""")
tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of images to process in a batch.""")

def loss(logits, labels):
    '''Compute loss
    Args:
        logits: logits tensor, [batch_size, n_classes]
        labels: one-hot labels
    '''
    with tf.name_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,name='cross-entropy')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar('loss', loss)
        return loss
    
def accuracy(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor,
    """
    with tf.name_scope('accuracy') as scope:
        correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)*100.0
        tf.summary.scalar('accuracy', accuracy)
    return accuracy



def num_correct_prediction(logits, labels):
    """Evaluate the quality of the logits at predicting the label.
    Return:
      the number of correct predictions
    """
    correct = tf.equal(tf.arg_max(logits, 1), tf.arg_max(labels, 1))
    correct = tf.cast(correct, tf.int32)
    n_correct = tf.reduce_sum(correct)
    return n_correct



def optimize(loss, learning_rate, global_step):
    '''optimization, use Gradient Descent as default
    '''
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op


def train():
    # pre_trained_weights = './/vgg16_pretrain//vgg16.npy'
    # data_dir = '../cifar10_data/cifar-10-batches-bin'
    train_log_dir = './tmp/VGG/train'
    val_log_dir = './tmp/VGG/val'

    # tra_image_batch, tra_label_batch = decode_tfrecord.read_cifar10(data_dir=data_dir,
    #                                              is_train=True,
    #                                              batch_size= BATCH_SIZE,
    #                                              shuffle=True)
    # val_image_batch, val_label_batch = decode_tfrecord.read_cifar10(data_dir=data_dir,
    #                                              is_train=False,
    #                                              batch_size= BATCH_SIZE,
    #                                              shuffle=False)
    tra_image_batch, tra_label_batch = decode_tfrecord.input('train', FLAGS.data_dir)
    val_image_batch, val_label_batch = decode_tfrecord.input('validation', FLAGS.data_dir)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 3])
    y_ = tf.placeholder(tf.int64, shape=[BATCH_SIZE])

    logits = VGG.VGG16N(x, N_CLASSES)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_, name='cross-entropy')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    loss_summary = tf.summary.scalar('loss', loss)

    correct = tf.equal(tf.arg_max(logits, 1), y_)
    correct = tf.cast(correct, tf.float32)
    accuracy = tf.reduce_mean(correct) * 100.0
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

    my_global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=my_global_step)

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    # summary_op = tf.summary.merge([loss_summary, accuracy_summary])

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # load the parameter file, assign the parameters, skip the specific layers
    # tools.load_with_skip(pre_trained_weights, sess, ['fc6','fc7','fc8'])


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    tra_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_: tra_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels})
                tra_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))

                summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    tra_summary_writer.close()
    val_summary_writer.close()


train()

