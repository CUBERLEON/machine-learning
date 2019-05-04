from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Reshape, UpSampling2D, Bidirectional, GRU, LSTM
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential


def custom_cifar_very_small_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=input_shape, name='conv_1', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten(name='features'))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def custom_cifar_small_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=input_shape, name='conv_1', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), name='conv_2', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten(name='features'))

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def custom_cifar_medium_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(16, (3, 3), input_shape=input_shape, name='conv_1', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), name='conv_2', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), name='conv_3', activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten(name='features'))

    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def custom_cifar_very_small_model_inv(input_shape, num_classes):
    model = Sequential()

    model.add(Reshape((16, 16, 16), input_shape=(4096,)))

    model.add(UpSampling2D(size=(2, 2), name='up_sample_2'))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, (3, 3), name='conv_2', padding='same'))

    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


def custom_cifar_small_model_inv(input_shape, num_classes):
    model = Sequential()

    model.add(Reshape((8, 8, 32), input_shape=(2048,)))

    model.add(UpSampling2D(size=(2, 2), name='up_sample_1'))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(16, (3, 3), name='conv_1', padding='same'))

    model.add(UpSampling2D(size=(2, 2), name='up_sample_2'))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, (3, 3), name='conv_2', padding='same'))

    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


def custom_cifar_medium_model_inv(input_shape, num_classes):
    model = Sequential()

    model.add(Reshape((4, 4, 64), input_shape=(1024,)))
    model.add(UpSampling2D(size=(2, 2), name='up_sample_1'))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(32, (3, 3), name='conv_1', padding='same'))

    model.add(UpSampling2D(size=(2, 2), name='up_sample_2'))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(16, (3, 3), name='conv_2', padding='same'))

    model.add(UpSampling2D(size=(2, 2), name='up_sample_3'))
    model.add(Activation('relu'))
    model.add(Conv2DTranspose(3, (3, 3), name='conv_3', padding='same'))
    model.add(Activation('sigmoid'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


def custom_cifar_gru_model(input_shape, num_classes):
    model = Sequential()

    model.add(Bidirectional(GRU(units=128,
                                activation='relu',
                                dropout=0.1,
                                recurrent_dropout=0.1,
                                return_sequences=True), input_shape=input_shape))
    model.add(GRU(units=128,
                  activation='relu',
                  dropout=0.1,
                  recurrent_dropout=0.1,
                  return_sequences=False))

    model.add(Dropout(0.1))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def custom_cifar_lstm_model(input_shape, num_classes):
    model = Sequential()

    model.add(LSTM(units=256,
                   activation='relu',
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   return_sequences=False,
                   input_shape=input_shape))

    model.add(Dropout(0.2))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def cifar_model_builder(type: str):
    if type == "very_small":
        return custom_cifar_very_small_model
    elif type == "small":
        return custom_cifar_small_model
    elif type == "medium":
        return custom_cifar_medium_model
    elif type == "gru":
        return custom_cifar_gru_model
    elif type == "lstm":
        return custom_cifar_lstm_model
    else:
        raise ValueError


def cifar_model_builder_inv(type: str):
    if type == "very_small":
        return custom_cifar_very_small_model_inv
    if type == "small":
        return custom_cifar_small_model_inv
    elif type == "medium":
        return custom_cifar_medium_model_inv
    else:
        raise ValueError
