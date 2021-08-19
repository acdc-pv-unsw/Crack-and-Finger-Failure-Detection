from keras.layers import Input
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np


class Autoencoder(object):
    def __init__(self, name, emb_size):
        self.name = name
        self.encoder = Sequential()
        self.decoder = Sequential()
        self.model = None
        self.emb_size = emb_size

    def autoencoder(self, img_shape):
        raise NotImplementedError()

    def fit(self, X_train, X_test, epochs=20):
        history = self.model.fit(x=X_train, y=X_train, epochs=epochs, validation_data=[X_test, X_test])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        return history

    def build(self, X):
        shape = X.shape[1:]

        self.autoencoder(shape)
        inp = Input(shape)
        code = self.encoder(inp)
        reconstruction = self.decoder(code)

        self.model = Model(inp, reconstruction)
        self.model.compile(optimizer='adamax', loss='mse')

        print(self.model.summary())

    def visualize(self, img, name):
        """Draws original, encoded and decoded images"""
        # img[None] will have shape of (1, 32, 32, 3) which is the same as the model input
        code = self.encoder.predict(img[None])[0]
        reco = self.decoder.predict(code[None])[0]

        plt.subplot(1, 3, 1)
        plt.title("Original")
        self.show(img)

        plt.subplot(1, 3, 2)
        plt.title("Code")
        plt.imshow(code.reshape([code.shape[-1] // 2, -1]))

        plt.subplot(1, 3, 3)
        plt.title("Reconstructed")
        self.show(reco)
        plt.savefig('results/{}.png'.format(name))
        plt.show()

    def show(self, img):
        plt.imshow(np.clip(img + 0.5, 0, 1))