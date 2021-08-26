from .Autoencoder import Autoencoder
import numpy as np
from keras.models import Model
from keras.layers import InputLayer, Conv2D, MaxPooling2D, UpSampling2D, Input


class ConvolutionalAutoencoder(Autoencoder):
    def __init__(self, emb_size, hidden_states, kernels, max_pool):
        super(ConvolutionalAutoencoder, self).__init__('Convolutional', emb_size)
        self.hiddens = hidden_states
        self.kernels = kernels
        self.max_pool = max_pool

    def autoencoder(self, img_shape):
        num_hidden = len(self.hiddens)

        self.encoder.add(InputLayer(img_shape))
        for i, hidden in enumerate(self.hiddens):
            self.encoder.add(Conv2D(hidden, self.kernels[i], activation='tanh', padding='same'))
            self.encoder.add(MaxPooling2D(self.max_pool[i], padding='same'))

        self.encoder.add(Conv2D(self.emb_size, self.kernels[-1], activation='tanh', padding='same'))
        self.encoder.add(MaxPooling2D(self.max_pool[-1], padding='same'))

        self.decoder.add(InputLayer((int(img_shape[0] / np.power(2, num_hidden+1)),
                                     int(img_shape[1] / np.power(2, num_hidden+1)), self.emb_size)))

        self.decoder.add(Conv2D(self.emb_size, self.kernels[-1], activation='tanh', padding='same'))
        self.decoder.add(UpSampling2D(self.max_pool[-1]))

        for i in range(len(self.hiddens)-1, -1, -1):
            self.decoder.add(Conv2D(self.hiddens[i], self.kernels[i], activation='tanh', padding='same'))
            self.decoder.add(UpSampling2D(self.max_pool[i]))

        self.decoder.add(Conv2D(img_shape[2], self.kernels[0], activation='tanh', padding='same'))
