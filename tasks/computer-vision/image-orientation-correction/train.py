from dataset import create_dataset
from model import OrientationModel

train_data_path = ''
validate_data_path = ''
log_path = ''

params = {'train_size': 0,
          'validation_size': 0,
          'shape': (512, 512, 2),
          'n_classes': 360,
          'batch_size': 32}

train_dataset = create_dataset(train_data_path,
                               params['train_size'],
                               params['shape'],
                               n_classes=params['n_classes'],
                               batch_size=params['batch_size'])

validation_dataset = create_dataset(validate_data_path,
                                    params['validation_size'],
                                    params['shape'],
                                    n_classes=params['n_classes'],
                                    batch_size=params['batch_size'])

model = OrientationModel(train_dataset, validation_dataset, params)
model.train(log_path)
