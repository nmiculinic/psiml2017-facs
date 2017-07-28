import dataset
import sys
import numpy as np
import keras
import os
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from haikunator import Haikunator
from PIL import Image, ImageDraw
from logging import FileHandler
import subprocess

haikunator = Haikunator()


class LandmarkPreview(Callback):
    def __init__(self, out_dir,
                 batch_size=10, validation_gen=None):
        super().__init__()
        self.out_dir = out_dir
        self.batch_size = batch_size
        self.validation_gen = validation_gen
        os.makedirs(out_dir, exist_ok=True)

    def draw_landmarks(self, image, landmarks, r=1, fill_color=(255,0,0,100)):
        draw = ImageDraw.Draw(image)
        for row in landmarks:
            x, y = row
            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill_color)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data:
            data = self.validation_data 
        else:
            x, y = next(self.validation_gen)
            data = [x, y, None]

        val_size = data[0].shape[0]
        samples = np.random.choice(np.arange(val_size), self.batch_size)

        for i, x, y_true, y_pred in zip(
                np.arange(val_size), 
                data[0][samples],
                data[1][samples],
                self.model.predict(data[0][samples]),
            ):
            img = Image.fromarray(np.squeeze(x, axis=2) * 255.0).convert("RGBA")
            # img.save(os.path.join(self.out_dir, "epoh_%02d_%02d_marker.png" % (epoch, i)))
            self.draw_landmarks(img, y_true, r=1, fill_color=(255,0,0,100))
            self.draw_landmarks(img, y_pred, r=1, fill_color=(0,255,0,100))
            img.save(os.path.join(self.out_dir, "epoh_%02d_%02d_marker.png" % (epoch, i)))


def complex_model(input_shape, l2_reg):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
                     kernel_regularizer=l2(l2_reg),
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv2D(
        32, 
        kernel_size=(3, 3),
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        padding='SAME',
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(
        64, 
        kernel_size=(3, 3),
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        padding='SAME',
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(
        128,
        kernel_size=(3, 3),
        activation='relu',
        kernel_regularizer=l2(l2_reg),
        padding='SAME',
    ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(
        128, 
        activation='relu',
        kernel_regularizer=l2(l2_reg),
    ))
    model.add(BatchNormalization())
    model.add(Dense(
        66 * 2, 
        activation='relu'
    ))
    model.add(Lambda(lambda x: x * input_shape[0]))
    model.add(Reshape((66, 2)))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
    )
    return model


def simple_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
                     input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
    ))
    model.add(MaxPooling2D())
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(66 * 2, activation='relu'))
    model.add(Lambda(lambda x: x * input_shape[0]))
    model.add(Reshape((66, 2)))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
    )
    return model


if __name__ == "__main__":
    log_fmt = '[%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    name = haikunator.haikunate() 
    logger = logging.getLogger(name)
    os.makedirs(os.path.join('.', 'logs', name), exist_ok=True)
    fh = FileHandler(os.path.join('.', 'logs', name, 'log.txt'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    logger.info("Git commit status %s", git_hash)
    logger.info("Started data loading.")
    dataset = dataset.Pain(sys.argv[1])
    # model = complex_model((124, 124, 1), 1e-3)
    model = simple_model((124, 124, 1))
    model.summary()
    logger.info("Model summary\n%s", model.to_json())
    
    checkpointer = ModelCheckpoint(
             os.path.join('.', 'logs', name, 'checkpoint'),
             verbose=0, 
             save_best_only=True
    )
    tensorboard = TensorBoard(
        os.path.join('.', 'logs', name),
        # write_grads=True,
        # histogram_freq=3
    )
    landmarks = LandmarkPreview(
        os.path.join('.', 'logs', name, 'pics'),
        validation_gen = dataset.test_generator(10)
    )

    model.fit_generator(
        generator=dataset.train_generator(32),
        steps_per_epoch=3000,
        epochs=30,
        verbose=1,
        validation_data=dataset.test_generator(32),
        validation_steps=1,
        callbacks = [checkpointer, tensorboard, landmarks],
        # max_queue_size = 100,
        use_multiprocessing=True
    )
    logger.info("Model training finished!")
    model.save(os.path.join('.', 'models', name + "_model.h5"))
    logger.info("Model saved")
