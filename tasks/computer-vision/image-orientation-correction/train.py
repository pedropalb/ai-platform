import os
import json

import click
# import mlflow
# import mlflow.keras

from dataset import create_dataset
from model import OrientationModel

# mlflow.set_experiment('orientation-correction')
# mlflow.keras.autolog()


@click.command()
@click.option('--config-path', required=True, help="The path to the configuration json.")
@click.option('--data-path', required=True, help="The path to the training and validation data.")
def main(config_path, data_path):
    with open(config_path) as f:
        configs = json.load(f)

    train_dataset = create_dataset(data_path + '/train.tfrecord',
                                   configs['train_data']['size'],
                                   configs['shape'],
                                   n_classes=configs['n_classes'],
                                   batch_size=configs['batch_size'])

    validation_dataset = create_dataset(data_path + '/validation.tfrecord',
                                        configs['validation_data']['size'],
                                        configs['shape'],
                                        n_classes=configs['n_classes'],
                                        batch_size=configs['batch_size'])

    model = OrientationModel(train_dataset,
                             validation_dataset,
                             configs,
                             use_imagenet=True)
    model.train('./train_logs')


if __name__ == '__main__':
    main()
