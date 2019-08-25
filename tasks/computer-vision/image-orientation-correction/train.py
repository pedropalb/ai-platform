import json

import click

from dataset import create_dataset
from model import OrientationModel


@click.command()
@click.option('--config-path', required=True, help="The path to the configuration json.")
def main(config_path):
    with open(config_path) as f:
        configs = json.load(f)

    train_dataset = create_dataset(configs['train_data']['tf_record_path'],
                                   configs['train_data']['size'],
                                   configs['shape'],
                                   n_classes=configs['n_classes'],
                                   batch_size=configs['batch_size'])

    validation_dataset = create_dataset(configs['validation_data']['tf_record_path'],
                                        configs['validation_data']['size'],
                                        configs['shape'],
                                        n_classes=configs['n_classes'],
                                        batch_size=configs['batch_size'])
    print(train_dataset)
    print(validation_dataset)

    # model = OrientationModel(train_dataset, validation_dataset, configs)
    # model.train(configs['log_path'])


if __name__ == '__main__':
    main()
