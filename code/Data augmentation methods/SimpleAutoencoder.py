from model.Autoencoder import Autoencoder
import numpy as np
from keras.layers import InputLayer, Dense, Flatten, Reshape


class SimpleAutoencoder(Autoencoder):
    def __init__(self, emb_size):
        super(SimpleAutoencoder, self).__init__('Simple', emb_size)

    def autoencoder(self, img_shape):
        # The encoder
        self.encoder.add(InputLayer(img_shape))
        self.encoder.add(Flatten())

        self.encoder.add(Dense(self.emb_size))

        # The decoder
        self.decoder.add(InputLayer((self.emb_size, )))
        self.decoder.add(Dense(np.prod(img_shape)))
        self.decoder.add(Reshape(img_shape))
