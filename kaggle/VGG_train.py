import os
import os.path

import numpy as np
import tensorflow as tf

import input_data
import VGG

import pandas as pd

IMG_W = 28
IMG_H = 28
N_CLASSES = 10
BATCH_SIZE = 32
learning_rate = 0.01
MAX_STEP = 3000 

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_log_dir', './kaggle/output/train',
                           """.""")
tf.app.flags.DEFINE_string('val_log_dir', './kaggle/output/val',
                           """.""")


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
    # train_log_dir = 'F:\\tmp\\buysell\\VGG_train\\train'
    # val_log_dir = 'F:\\tmp\\buysell\\VGG_train\\val'

    # tra_image_batch, tra_label_batch = decode_tfrecord.input('train', FLAGS.data_dir)
    # val_image_batch, val_label_batch = decode_tfrecord.input('validation', FLAGS.data_dir)

    df = pd.read_csv('./kaggle/train.csv', sep=',')
    train_df = df.sample(frac=0.7,random_state=123)
    val_df = df[~df.index.isin(train_df.index)]

    train_batch = input_data.get(train_df, BATCH_SIZE)
    val_batch = input_data.get(val_df, BATCH_SIZE)

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_W, IMG_H, 1])
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
    tra_summary_writer = tf.summary.FileWriter(FLAGS.train_log_dir, sess.graph)
    val_summary_writer = tf.summary.FileWriter(FLAGS.val_log_dir, sess.graph)

    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run(train_batch)
            _, tra_loss, tra_acc = sess.run([train_op, loss, accuracy],
                                            feed_dict={x: tra_images, y_: tra_labels})
            if step % 50 == 0 or (step + 1) == MAX_STEP:
                print('Step: %d, loss: %.4f, accuracy: %.4f%%' % (step, tra_loss, tra_acc))
                summary_str = sess.run(summary_op, feed_dict={x: tra_images, y_: tra_labels})
                tra_summary_writer.add_summary(summary_str, step)

            if step % 200 == 0 or (step + 1) == MAX_STEP:
                val_images, val_labels = sess.run(val_batch)
                val_loss, val_acc = sess.run([loss, accuracy],
                                             feed_dict={x: val_images, y_: val_labels})
                print('**  Step %d, val loss = %.2f, val accuracy = %.2f%%  **' % (step, val_loss, val_acc))

                summary_str = sess.run(summary_op, feed_dict={x: val_images, y_: val_labels})
                val_summary_writer.add_summary(summary_str, step)

            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(FLAGS.train_log_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    tf.train.write_graph(sess.graph_def, FLAGS.train_log_dir, "vgg_model.pb", as_text=False)

    coord.join(threads)
    sess.close()
    tra_summary_writer.close()
    val_summary_writer.close()


train()

