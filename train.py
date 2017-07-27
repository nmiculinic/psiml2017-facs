import dataset
import sys
import numpy as np
import keras
import os
import logging
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback
from haikunator import Haikunator
from PIL import Image, ImageDraw


haikunator = Haikunator()


class LandmarkPreview(Callback):
    def __init__(self, out_dir,
                 batch_size=32,):
        super().__init__()
        self.out_dir = out_dir
        self.batch_size = batch_size
        os.makedirs(out_dir, exist_ok=True)

    def draw_landmarks(self, image, landmarks, r=1, fill_color=(255,0,0,100)):
        draw = ImageDraw.Draw(image)
        for row in landmarks:
            x, y = row
            draw.ellipse((x-r, y-r, x+r, y+r), fill=fill_color)
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.validation_data:
            val_size = self.validation_data[0].shape[0]
            samples = np.random.choice(np.arange(val_size), self.batch_size)

            x_test = self.validation_data[0][samples]
            y_targets = self.validation_data[1][samples]
            y_preds = self.model.predict(x_test)

            for i, x, y_true, y_pred in zip(np.arange(val_size), x_test, y_targets, y_preds):
                img = Image.fromarray(np.squeeze(x, axis=2) * 255.0).convert("RGBA")
                # img.save(os.path.join(self.out_dir, "epoh_%2d_%2d.png" % (epoch, i)))
                self.draw_landmarks(img, y_true, r=2, fill_color=(255,0,0,100))
                self.draw_landmarks(img, y_pred, r=2, fill_color=(0,255,0,100))
                img.save(os.path.join(self.out_dir, "epoh_%2d_%2d_marker.png" % (epoch, i)))

if __name__ == "__main__":
    name = haikunator.haikunate() 
    print(name)
    log_fmt = '[%(levelname)s] %(name)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(name)
    logger.info("Started data loading.")

    sol = dataset.cohn_kanade_dataset(sys.argv[1])
    for x in sol:
        x['image'] = np.array(x['image'].convert("L"))
        
    sol = list(filter(lambda x:x['image'].shape == (490, 640), sol))

    x_train = [x['image'] for x in sol]
    x_train = np.array(x_train) / 255.0
    x_train = x_train[:, :, :, None]

    y_train = np.array([x['landmarks'] for x in sol])

    print(x_train.shape, y_train.shape)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
                     input_shape=x_train[0].shape))
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
    ))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
    ))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
    ))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(68 * 2, activation='relu'))
    model.add(Reshape((68, 2)))
    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
    )

    
    checkpointer = ModelCheckpoint(
             os.path.join('.', 'logs', name, 'checkpoint'),
             verbose=0, 
             save_best_only=True
    )

    tensorboard = TensorBoard(
        os.path.join('.', 'logs', name),
        write_grads=True,
        histogram_freq=3
    )

    landmarks = LandmarkPreview(
        os.path.join('.', 'logs', name, 'pics'),
        32
    )

    model.fit(x_train, y_train,
        batch_size=16,
        epochs=20,
        verbose=1,
        validation_split=0.1,
        callbacks = [checkpointer, tensorboard, landmarks]
    )
    model.save(os.path.join('.', 'models', name + "_model.h5"))
