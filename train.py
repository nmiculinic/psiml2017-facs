import dataset
import sys
import numpy as np

if __name__ == "__main__":
    sol = dataset.cohn_kanade_dataset(sys.argv[1])
    
    sol = list(filter(lambda x:x['image'].shape == (490, 640), sol))

    x_train = np.array([x['image'] for x in sol]) / 255.0
    y_train = np.array([x['landmarks'] for x in sol])
    x_train = x_train[:, :, :, None]
    print(x_train.shape, y_train.shape)
    print(x_train[0])
    print(y_train[0])

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Reshape
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    from keras.callbacks import ModelCheckpoint

    model = Sequential()
    model.add(Conv2D(2, kernel_size=(3, 3),
                     activation='relu',
                     padding='SAME',
                     input_shape=x_train[0].shape))
    model.add(Flatten())
    model.add(Dense(68 * 2, activation='relu'))
    model.add(Reshape((68, 2)))

    model.compile(loss=keras.losses.mean_squared_error,
                  optimizer='adam',
    )

    
    checkpointer = ModelCheckpoint(
            'test.chkp',
             verbose=0, 
             save_best_only=True
    )
    model.fit(x_train, y_train,
              batch_size=2,
              epochs=1,
              verbose=1,
              validation_split=0.1,
              callbacks = [checkpointer]
    )

    model.save("exit.h5")
