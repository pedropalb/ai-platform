from pathlib import Path

import click
import tensorflow as tf


def generate_example(jpeg_string, label):
    jpeg_feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=[jpeg_string]))
    label_feature = tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))

    example = tf.train.Example(features=tf.train.Features(
        feature={
            'encoded': jpeg_feature,
            'angle': label_feature
        }
    ))

    return example.SerializeToString()


@click.command()
@click.option("--input-dir", required=True, help="The root dir containing the images.")
@click.option("--output-path", default="data.tfrecord", show_default=True, help="The TFRecord file path.")
def generate_tf_record(input_dir, output_path):
    input_dir = Path(input_dir).resolve()
    output_path = Path(output_path).resolve()

    with tf.io.TFRecordWriter(str(output_path)) as writer:
        for file_path in input_dir.iterdir():
            with open(str(file_path), 'rb') as f:
                jpeg_string = f.read()
                label = int(file_path.name.split('_')[0])

                serialized_example = generate_example(jpeg_string, label)

                writer.write(serialized_example)


if __name__ == '__main__':
    generate_tf_record()
