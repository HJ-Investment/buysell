import tensorflow as tf
import os
import cv2

# tf.app.flags.DEFINE_integer(
#     'batch_size', 32, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_readers', 4,
    'The number of parallel readers that read data from the dataset.')


tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_integer(
    'train_image_size_width', 256, 'Train image size')

tf.app.flags.DEFINE_integer(
    'train_image_size_height', 256, 'Train image size')

FLAGS = tf.app.flags.FLAGS
slim = tf.contrib.slim
LABELS_FILENAME = 'label.txt'

_FILE_PATTERN = 'macd_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 23100, 'validation': 9900}

_NUM_CLASSES = 3

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying size.',
    'label': 'A single integer between 0 and 4',
}

def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
        dataset_dir: The directory in which the labels file is found.
        filename: The filename where the class names are written.

    Returns:
        A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    print(lines)
    lines = filter(None, lines)
    print(lines)

    labels_to_class_names = {}
    for index, line in enumerate(lines):
        print(line)
        # index = line.index(':')
        labels_to_class_names[index] = line
    return labels_to_class_names

def input(split_name, dataset_dir, file_pattern=None, reader=None):

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    # 第一步
    # 将example反序列化成存储之前的格式。由tf完成
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }
    # 第一步
    # 将反序列化的数据组装成更高级的格式。由slim完成
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded','image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }
    # 解码器，进行解码
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    
    labels_to_names = read_label_file(dataset_dir)

    # dataset对象定义了数据集的文件位置，解码方式等元信息
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=decoder,
        num_samples=SPLITS_TO_SIZES[split_name],#训练数据的总数
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=_NUM_CLASSES,
        labels_to_names=labels_to_names #字典形式，格式为：id:class_call,
        )
    # provider对象根据dataset信息读取数据
    provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,
            num_readers=FLAGS.num_readers,
            common_queue_capacity=20 * FLAGS.batch_size,
            common_queue_min=10 * FLAGS.batch_size)

    # 获取数据，获取到的数据是单个数据，还需要对数据进行预处理，组合数据
    [image, label] = provider.get(['image', 'label'])
    # 图像预处理
    # image = image_preprocessing_fn(image, train_image_size, train_image_size)
    image = tf.image.resize_images(image, [FLAGS.train_image_size_width, FLAGS.train_image_size_height])

    images, labels = tf.train.batch(
                [image, label],
                batch_size=FLAGS.batch_size,
                num_threads=FLAGS.num_preprocessing_threads,
                capacity=5 * FLAGS.batch_size)
    # labels = slim.one_hot_encoding(
    #             labels, dataset.num_classes - FLAGS.labels_offset)
    batch_queue = slim.prefetch_queue.prefetch_queue(
                [images, labels], capacity=2)
    # 组好后的数据
    images, labels = batch_queue.dequeue()
    
    return images, labels

# with tf.Session() as sess:
#     labels = input('train', 'F:\\Code\\buysell\\data\\tfrecords')
#     threads = tf.train.start_queue_runners(sess=sess)
#     # 变量初始化
#     sess.run(tf.global_variables_initializer())

#     for i in range(5):
#         imgs, labs = sess.run(labels)
#         # print(labs)
#         for img in imgs:
#             cv2.imwrite('F:\Code\\buysell\data\pic_data\\raw_test\\%d.png' % i, img)
