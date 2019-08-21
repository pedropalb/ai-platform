import functools

import tensorflow as tf


def create_dataset(file_path, size, shape, n_classes, batch_size):
    dataset = tf.data.TFRecordDataset(file_path)

    decode = functools.partial(shape, n_classes)
    dataset = dataset.map(decode)
    dataset = dataset.map(normalize)

    dataset = dataset.shuffle(size)

    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def decode(shape, n_classes, serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'encoded': tf.FixedLenFeature([], tf.string),
            'angle': tf.FixedLenFeature([], tf.int64),
        })

    image = tf.image.decode_jpeg(features['encoded'], channels=3)
    image.set_shape(shape)

    label = tf.cast(features['angle'], tf.int32)

    return image, tf.one_hot(label, n_classes)


def normalize(image, label):
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    return image, label



