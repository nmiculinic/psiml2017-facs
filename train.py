import dataset
import sys
import numpy as np
import keras
import os
import logging
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization, Lambda
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping, ReduceLROnPlateau
from haikunator import Haikunator
from PIL import Image, ImageDraw
from logging import FileHandler
import subprocess
import argparse

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
            w, h = img.size
            # img = img.resize((2*w, 2*h), resample=Image.BICUBIC)
            # img.save(os.path.join(self.out_dir, "epoh_%02d_%02d_marker.png" % (epoch, i)))
            self.draw_landmarks(img, y_true, r=1, fill_color=(255,0,0,100))
            self.draw_landmarks(img, y_pred, r=1, fill_color=(0,255,0,100))
            img.save(os.path.join(self.out_dir, "epoh_%02d_%02d_marker.png" % (epoch, i)))


def complex_model(input_shape, l2_reg, layers, act='relu'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5),
                     activation=act,
                     padding='SAME',
                     kernel_regularizer=l2(l2_reg),
                     input_shape=input_shape))
    for i, num in enumerate(layers):
        for _ in range(num):
            model.add(Conv2D(
                32 * (2**i), 
                kernel_size=(3, 3),
                activation='relu',
                kernel_regularizer=l2(l2_reg),
                padding='SAME',
            ))
            if act == 'relu':
                model.add(BatchNormalization())
        model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(
        48, 
        kernel_regularizer=l2(l2_reg),
    ))
    model.add(Dense(
        128, 
        activation='relu',
        kernel_regularizer=l2(l2_reg),
    ))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(
        66 * 2, 
        activation='relu'
    ))
    model.add(Lambda(lambda x: x * 80.0))
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
    model.add(Lambda(lambda x: 80.0 * x))
    model.add(Reshape((66, 2)))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
    )
    return model


def train_model(name, model, model_kwargs):
    logger = logging.getLogger(name)
    os.makedirs(os.path.join('.', 'logs', name), exist_ok=True)
    fh = FileHandler(os.path.join('.', 'logs', name, 'log.txt'))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"])
    logger.info("Git commit status %s", git_hash)
    logger.info("Args\n%s", dataset.pp.pformat(model_kwargs))
    logger.info("Started data loading.")
    ds = dataset.Pain(
        args.dataset_path, 
        picture_size=args.picture_size, 
        crop_window=args.crop_window,
        max_angle=args.max_angle
    )

    logger.info("Model summary")
    model.summary(print_fn=lambda x: logger.info(str(x)))
    
    checkpointer = ModelCheckpoint(
             os.path.join('.', 'logs', name, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
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
        validation_gen = ds.test_generator(10)
    )
    early_stop = EarlyStopping(
        monitor='val_loss', 
        min_delta=0.1, 
        patience=70, 
        verbose=1, 
        mode='min'
    )

    reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', 
            factor=0.2,
            patience=30, 
            min_lr=1e-6,
            verbose=1
    )

    model.save("/tmp/test_save_keras")
    load_model("/tmp/test_save_keras")
    logger.info("Successfully tested save/load")
    try:
        model.fit_generator(
            generator=ds.train_generator(args.batch_size),
            steps_per_epoch=args.steps,
            epochs=args.epochs,
            verbose=1,
            validation_data=ds.test_generator(10),
            validation_steps=1,
            callbacks = [checkpointer, tensorboard, landmarks, early_stop, reduce_lr],
            # max_queue_size = 100,
            use_multiprocessing=True,
            workers=args.num_workers
        )
        logger.info("Model training finished!")
    except KeyboardInterrupt:
        logger.info("Model training interrupted!!!")

    model.save(os.path.join('.', 'models', name + "_model.h5"))
    logger.info("Model saved")

if __name__ == "__main__":
    log_fmt = '[%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    args = argparse.ArgumentParser()
    args.add_argument("dataset_path")
    args.add_argument("--name", default=haikunator.haikunate())
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--epochs", type=int, default=1000)
    args.add_argument("--steps", type=int, default=100, help="Steps per epoh")
    args.add_argument("--picture_size", type=int, default=128)
    args.add_argument("--crop_window", type=int, default=10)
    args.add_argument("--max_angle", type=float, default=15.0)
    args.add_argument("--num_workers", type=int, default=20)
    args.add_argument("--test", action="store_true")
    args = args.parse_args()

    name = args.name
    # if args.test:
    #     model = simple_model(
    # else:
    #     model = complex_model((args.picture_size, args.picture_size, 1), 1e-2)

    i = 0
    while True:
        i += 1
        act = random.choice(['relu', 'selu'])
        num_layers = random.randint(2, 4)
        layers = [random.randint(1, 3) for _ in range(num_layers)]
        reg = 10 ** (-5 * random.random())
        kwargs = {
            'l2_reg': reg,
            'act': act,
            'layers': layers,
            'input_shape': (args.picture_size, args.picture_size, 1)
        }

        model = complex_model(**kwargs)
        train_model(args.name + "_%d" % i, model, kwargs) 
