import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, Dense, Input, Conv2DTranspose, Flatten, Reshape
from tensorflow.python.keras.models import Model

from common.config import MODELS_DATA_DIR

img_height, img_width = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X = np.expand_dims(x_train, axis=-1)
print(X.shape)

input_image = Input(shape=(img_height, img_width, 1), name='image_imput')
x = Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_conv1', activation='relu')(
    input_image)
x = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', name='encoder_conv2', activation='relu')(x)
x = Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='encoder_conv3', activation='relu')(x)
x = Flatten()(x)
encoded = Dense(units=10)(x)

y = Dense(units=1152, activation='relu')(encoded)
y = Reshape((3, 3, 128))(y)
y = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='valid', name='decoder_deconv1',
                    activation='relu')(y)
y = Conv2DTranspose(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_deconv2',
                    activation='relu')(y)
decoded_image = Conv2DTranspose(filters=1, kernel_size=(5, 5), strides=(2, 2), padding='same', name='decoder_deconv3',
                                activation='relu')(y)

model = Model(inputs=input_image, outputs=decoded_image, name='CAE')

model_dir = MODELS_DATA_DIR / 'autoencoder_mnist'
tb = TensorBoard(log_dir='logs', write_graph=True)
mc = ModelCheckpoint(filepath=model_dir / 'top_weights.h5', monitor='acc', save_best_only='True',
                     save_weights_only='True', verbose=1)
es = EarlyStopping(monitor='loss', patience=15, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss')
callbacks = [tb, mc, es, rlr]
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.load_weights(str(model_dir / 'top_weights.h5'))
# model.fit(X, X, epochs=1000, batch_size=256, callbacks=callbacks)

np.random.shuffle(x_test)

fig = plt.figure(figsize=(20, 20))
fig_r = plt.figure(figsize=(20, 20))
for i in range(5):
    img = x_test[i]
    ax = fig.add_subplot(1, 5, i + 1)
    ax.set_title(f'{i + 1}')
    ax.imshow(img, cmap='gray')
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    result = np.squeeze(model.predict(img))
    ax_ = fig_r.add_subplot(1, 5, i + 1)
    ax_.set_title(f'{i + 1}')
    ax_.imshow(result, cmap='gray')
plt.show()
