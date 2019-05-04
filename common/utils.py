import inspect
import time
from pathlib import Path

import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import get_file

from common.plot import plot_curves
from common.tf_utils import TrainValTensorBoard


def download(file_path: Path, target_dir: Path):
    print(f'Downloading "{file_path}"')
    return Path(get_file(fname=str(file_path.name), origin=str(file_path), cache_subdir=str(target_dir.absolute())))


def extract(file_path: Path):
    data_dir = file_path.with_suffix('')

    if not data_dir.exists():
        print(f'Extracting "{file_path}"')
        get_file(fname=str(file_path.name), origin=str(file_path.as_uri()), extract=True, cache_subdir=str(data_dir.parent.absolute()))

    return data_dir


def keras_model(model, model_dir: Path,
                *, train: bool = False,
                restore: bool = True,
                train_data: tuple = None,
                batch_size: int = 64,
                epochs: int = 100,
                save_period: int = 10,
                early_stopping_patience: int = 10):

    model_dir.mkdir(exist_ok=True)

    tb = TrainValTensorBoard(log_dir=str(model_dir / f'logs-{time.strftime("%d_%m_%Y-%H_%M_%S")}'),
                             write_graph=True, write_grads=True, write_images=True)

    weigths_best_path = model_dir / 'weights.best.hdf5'

    mc = ModelCheckpoint(filepath=str(model_dir / 'weights.epoch_{epoch:02d}.val_loss_{val_loss:.2f}.hdf5'),
                         monitor='val_loss', verbose=1, period=save_period)
    mc_best = ModelCheckpoint(filepath=str(weigths_best_path),
                              monitor='val_loss', save_best_only='True', verbose=1)
    es = EarlyStopping(monitor='val_loss', patience=early_stopping_patience, verbose=1)

    if restore and not train and not weigths_best_path.exists():
        raise FileNotFoundError(weigths_best_path)
    if weigths_best_path.exists() and restore:
        model.load_weights(str(weigths_best_path))

    if train:
        x_train, x_val, y_train, y_val = train_data

        history = model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=[tb, mc, mc_best, es],
                  batch_size=batch_size, epochs=epochs, shuffle=True)

        plot_curves([history.history['acc'], history.history['val_acc']], title='Model accuracy',
                    xlabel='Epoch', ylabel='Accuracy', legend=['Train', 'Validation'])
        plot_curves([history.history['loss'], history.history['val_loss']], title='Model loss',
                    xlabel='Epoch', ylabel='Loss', legend=['Train', 'Validation'])


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]


def layer_functor(model, layer_name, learning_phase=0):
    functor = keras.backend.function([model.input, keras.backend.learning_phase()],
                                     [model.get_layer(layer_name).output])
    return lambda x: functor([x, learning_phase])[0]
