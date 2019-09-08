import tensorflow as tf
from keras import Input, Model
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Flatten, Dense
import keras.backend as K
from keras.optimizers import Adam

from callbacks import EvaluateInputTensor, TensorBoardWithImages


class OrientationModel:
    def __init__(self, train_dataset, validation_dataset, params, use_imagenet=False):
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.params = params

        self.train_model = self.__build_model(train_dataset, use_imagenet)
        self.validation_model = self.__build_model(validation_dataset, use_imagenet)

    def train(self, log_path):
        callbacks = self.__build_calbacks(log_path)

        sess = K.get_session()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        self.train_model.fit(epochs=self.params['n_epochs'],
                             steps_per_epoch=self.params['steps_per_epoch'],
                             callbacks=[EvaluateInputTensor(self.validation_model, steps=50)] + callbacks)

        self.train_model.save_weights('model_weights.h5')

        coord.request_stop()
        coord.join(threads)
        K.clear_session()

    def predict(self):
        pass

    def __calculate_error(self, y_true, y_pred):
        diff = 180 - abs(abs(K.argmax(y_true) - K.argmax(y_pred)) - 180)
        return K.mean(K.cast(K.abs(diff), K.floatx()))

    def __build_calbacks(self, logs_path):
        monitor = 'val_loss'
        checkpointer = ModelCheckpoint(
            filepath='./model.h5',
            monitor=monitor,
            save_best_only=True,
            mode='min',
            verbose=1
        )

        tensorboard = TensorBoardWithImages(logs_path, self.train_dataset, self.params['batch_size'])

        return [checkpointer, tensorboard]

    def __build_model(self, dataset, use_imagenet=False):
        weights = 'imagenet' if use_imagenet else None

        input = Input(tensor=dataset[0], shape=self.params['shape'])
        base_model = ResNet50(weights=weights, include_top=False)(input)

        x = Flatten()(base_model)
        x = Dense(512, name='fc2', activation='relu')(x)

        output = Dense(self.params['n_classes'], activation='softmax', name='fc360')(x)

        model = Model(inputs=input, outputs=output)

        optimizer = Adam(lr=0.01, decay=1e-3)
        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizer,
                      metrics=[self.__calculate_error],
                      target_tensors=[dataset[1]])

        return model
