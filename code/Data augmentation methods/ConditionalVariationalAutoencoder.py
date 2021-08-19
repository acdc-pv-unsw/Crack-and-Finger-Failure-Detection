from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras
from keras import losses
from keras.optimizers import SGD
import pandas as pd
from tensorflow.python.keras.layers import Embedding, Flatten, BatchNormalization


class ConditionalVariationalAutoencoder:
    def __init__(self, emb_size, hidden_states, y_dim, x_dim, batch_size=20):
        self.hiddens = hidden_states
        self.y_dim = y_dim
        self.x_dim = x_dim
        self.batch_size = batch_size
        self.emb_size = emb_size
        self.c_vae = None
        self.mu = None
        self.sigma = None
        np.random.seed(0)

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.emb_size),
                                  mean=0., stddev=1.)
        return z_mean + K.exp(z_log_sigma / 2) * epsilon

    def build(self):
        in_label = Input(shape=(self.y_dim,))

        in_label1 = Embedding(self.y_dim, 50)(in_label)

        input = Input(shape=(self.x_dim,))
        in_label2 = Flatten()(in_label1)
        merge = K.concatenate([input, in_label2])
        d1 = Dense(self.hiddens, activation='relu')(merge)
        b1 = BatchNormalization()(d1)
        d2 = Dense(self.hiddens / 2, activation='relu')(b1)
        b2 = BatchNormalization()(d2)
        d2_2 = Dense(self.hiddens / 2, activation='relu')(b2)
        b3 = BatchNormalization()(d2_2)
        d3 = Dense(self.hiddens / 4, activation='relu')(b3)
        b4 = BatchNormalization()(d3)
        d4 = Dense(self.hiddens / 4, activation='relu')(b4)
        b5 = BatchNormalization()(d4)
        d5 = Dense(self.hiddens / 8, activation='relu')(b5)
        b6 = BatchNormalization()(d5)
        d6 = Dense(self.hiddens / 8, activation='relu')(b6)
        b7 = BatchNormalization()(d6)

        z_mean = Dense(self.emb_size, activation='linear', name='mu')(b7)
        z_log_sigma = Dense(self.emb_size, activation='linear', name='sigma')(b7)
        z = Lambda(self.sampling)([z_mean, z_log_sigma])
        z_cond = K.concatenate([z, in_label], axis=-1)

        # Create encoder
        self.encoder = Model([input, in_label], [z_mean, z_log_sigma, z_cond], name='encoder')
        print(self.encoder.summary())
        # Create decoder
        latent_inputs = Input(shape=(self.emb_size + self.y_dim,), name='z_sampling')
        d1 = Dense(self.hiddens / 8, activation='relu')(latent_inputs)
        b1 = BatchNormalization()(d1)
        d2 = Dense(self.hiddens / 8, activation='relu')(b1)
        b2 = BatchNormalization()(d2)
        d3 = Dense(self.hiddens / 4, activation='relu')(b2)
        b3 = BatchNormalization()(d3)
        d4 = Dense(self.hiddens / 4, activation='relu')(b3)
        b4 = BatchNormalization()(d4)
        d5 = Dense(self.hiddens / 2, activation='relu')(b4)
        b5 = BatchNormalization()(d5)
        d6 = Dense(self.hiddens / 2, activation='relu')(b5)
        b6 = BatchNormalization()(d6)

        outputs = Dense(self.x_dim, activation='sigmoid')(b6)
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        print(self.decoder.summary())
        # instantiate VAE model
        outputs = self.decoder(self.encoder([input, in_label])[2])  # [2] is z_cond from encoder output list
        self.c_vae = Model([input, in_label], outputs, name='cond_vae')
        print(self.c_vae.summary())

        self.mu = z_mean
        self.sigma = z_log_sigma

        # mae = self.mae(y_true=x_inp, y_pred=outputs)
        # loss = self.loss(y_true=x_inp, y_pred=outputs)
        # mse = self.mse(y_true=x_inp, y_pred=outputs)
        c_vae_loss = self.vae_loss(y_true=input, y_pred=outputs)
        kl = self.kl_loss()
        r_loss = self.recon_loss(y_true=input, y_pred=outputs)

        self.c_vae.add_loss(c_vae_loss)
        self.c_vae.add_metric(kl, name='KL')
        self.c_vae.add_metric(r_loss, name='Rec_Loss')

        # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        #adam = tf.keras.optimizers.Adam(learning_rate=0.00001)

        self.c_vae.compile(optimizer='adam')

    def fit(self, X_train, X_test, n_epochs=20):
        x_train, y_train = np.asarray(X_train[0]).astype(np.float32), X_train[1]
        x_test, y_test = np.asarray(X_test[0]).astype(np.float32), X_test[1]

        y_train = np.asarray([np.asarray(ele) for ele in y_train]).astype(np.float32)
        y_test = np.asarray([np.asarray(ele) for ele in y_test]).astype(np.float32)

        cvae_hist = self.c_vae.fit([x_train, y_train], x_train, batch_size=self.batch_size, epochs=n_epochs,
                                   validation_data=([x_test, y_test], x_test),
                                   validation_split=0.1,
                                   callbacks=[EarlyStopping(patience=20)])
        return cvae_hist

    def mae(self, y_true, y_pred):
        mae = tf.losses.MeanAbsoluteError()
        return mae(y_true, y_pred)

    def mse(self, y_true, y_pred):
        return tf.losses.MSE(y_true, y_pred)

    def vae_loss(self, y_true, y_pred):
        # E[log P(X|z)]
        # recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
        recon = keras.metrics.binary_crossentropy(y_true, y_pred)
        # D_KL(Q(z|X) || P(z|X))
        # kl = 0.5 * K.sum(K.exp(self.sigma) + K.square(self.mu) - 1. - self.sigma, axis=-1)
        kl = -5e-4 * K.mean(1 + self.sigma - K.square(self.mu) - K.exp(self.sigma), axis=-1)
        return K.mean(recon + kl)

    def kl_loss(self):
        # return 0.5 * K.sum(K.exp(self.sigma) + K.square(self.mu) - 1. - self.sigma, axis=-1)
        return -5e-4 * K.mean(1 + self.sigma - K.square(self.mu) - K.exp(self.sigma), axis=-1)

    def recon_loss(self, y_true, y_pred):
        # return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
        return keras.metrics.binary_crossentropy(y_true, y_pred)

    def loss(self, y_true, y_pred):
        weights = K.cast(K.abs(K.argmax(y_true, axis=1) - K.argmax(y_pred, axis=1)) / K.int_shape(y_pred[1] - 1),
                         dtype='float32')
        return (1.0 - weights) * losses.categorical_crossentropy(y_true, y_pred)

    def plot_latent_space(self, x, y, name):
        x = np.asarray(x.values).astype(np.float32)
        y = np.asarray([np.asarray(ele) for ele in y]).astype(np.float32)
        encoded = self.encoder.predict([x, y])[2]
        plt.figure(figsize=(6, 6))
        plt.scatter(encoded[:, 0], encoded[:, 1], c=y[:, 0])
        plt.colorbar()
        plt.savefig('results/{}_latent_space.jpg'.format(name))
        plt.show()

    def generate_samples(self, label):
        z_sample = np.random.rand(1, 6)
        zs = K.constant(label, shape=[1, 2])
        z_sample = K.constant(z_sample, shape=[1, 6])
        x_decoded = self.decoder.predict(K.concatenate([z_sample, zs]))
        np.set_printoptions(formatter={'float': lambda x: "{0:0.6f}".format(x)})
        print(z_sample, zs)
        print(x_decoded)
        if label == [1.0, 0.0]:
            a = 0
        else:
            a = 1
        df = pd.DataFrame(x_decoded, columns=['ICA', 'kur', 'skew', 'sp', 'md',
                                              'sd', 'var', '25p', '75p', 'fw', 'kstat', 'entropy'])
        df['labels'] = a
        return df

    def plot_training(self):
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        plt.plot(self.c_vae.history.history['loss'])
        ax1.set_title('loss')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(self.c_vae.history.history['val_loss'])
        ax2.set_title('validation loss')
        plt.show()

    def save(self):
        model = self.c_vae
        model.save('cvae_model.h5')
