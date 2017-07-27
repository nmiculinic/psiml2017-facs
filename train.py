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
from keras.callbacks import ModelCheckpoint, TensorBoard
from haikunator import Haikunator


haikunator = Haikunator()
logging.basicConfig()

if __name__ == "__main__":
    name = haikunator.haikunate() 
    logger = logging.getLogger(name)
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

    model.fit(x_train, y_train,
        batch_size=16,
        epochs=1,
        verbose=1,
        validation_split=0.1,
        callbacks = [checkpointer, tensorboard]
    )
    model.save(os.path.join('.', 'models', name + "_model.h5"))
