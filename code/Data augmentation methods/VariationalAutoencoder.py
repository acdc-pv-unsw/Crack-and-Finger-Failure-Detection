from model.Autoencoder import Autoencoder
from keras import backend as K
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.callbacks import EarlyStopping
import numpy as np
from keras.utils import np_utils
import matplotlib.pyplot as plt


class VariationalAutoencoder(Autoencoder):
    def __init__(self, emb_size, hidden_states, y_dim, batch_size=20):
        super(VariationalAutoencoder, self).__init__('Variational', emb_size)
        self.hiddens = hidden_states
        self.y_dim = y_dim
        self.batch_size = batch_size
        self.c_vae = None
        self.mu = None
        self.sigma = None
        np.random.seed(0)

    def autoencoder(self, img_shape):
        pass

    def sampling(self, args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.emb_size),
                                  mean=0., stddev=0.1)
        return z_mean + K.exp(z_log_sigma/2) * epsilon

    def build(self, x_train):
        m = None # = self.batch_size
        x_dim = x_train.shape[1]
        x_inp = Input(batch_shape=(m, x_dim))
        cond = Input(batch_shape=(m, self.y_dim))
        inputs = K.concatenate([x_inp, cond], axis=-1)
        h = Dense(self.hiddens, activation='relu')(inputs)
        z_mean = Dense(self.emb_size)(h)
        z_log_sigma = Dense(self.emb_size)(h)
        z = Lambda(self.sampling)([z_mean, z_log_sigma])
        z_cond = K.concatenate([z, cond], axis=-1)

        # Create encoder
        self.encoder = Model([x_inp, cond], [z_mean, z_log_sigma, z_cond], name='encoder')
        print(self.encoder.summary())
        # Create decoder
        latent_inputs = Input(batch_shape=(m, self.emb_size+self.y_dim), name='z_sampling')
        h = Dense(self.hiddens, activation='relu')(latent_inputs)
        outputs = Dense(x_dim, activation='sigmoid')(h)
        self.decoder = Model(latent_inputs, outputs, name='decoder')
        print(self.decoder.summary())
        # instantiate VAE model
        outputs = self.decoder(self.encoder([x_inp, cond])[2])
        self.c_vae = Model([x_inp, cond], outputs, name='cond_vae_mlp')
        print(self.c_vae.summary())
        
        self.mu = z_mean
        self.sigma = z_log_sigma

        c_vae_loss = self.vae_loss(y_true=x_inp, y_pred=outputs)
        kl = self.kl_loss()
        r_loss = self.recon_loss(y_true=x_inp, y_pred=outputs)
    
        self.c_vae.add_loss(c_vae_loss)
        self.c_vae.add_metric(kl, name='KL')
        self.c_vae.add_metric(r_loss, name='Rec_Loss')

        self.c_vae.compile(optimizer='adam')

    def fit(self, X_train, X_test, n_epochs=20):
        x_train, y_train = np.asarray(X_train[0].values).astype(np.float32), X_train[1].values
        x_test, y_test = np.asarray(X_test[0].values).astype(np.float32), X_test[1].values

        y_train = np.asarray([np.asarray(ele) for ele in y_train]).astype(np.float32)
        y_test = np.asarray([np.asarray(ele) for ele in y_test]).astype(np.float32)

        cvae_hist = self.c_vae.fit([x_train, y_train], x_train, batch_size=self.batch_size, epochs=n_epochs,
                             validation_data=([x_test, y_test], x_test),
                             callbacks=[EarlyStopping(patience=10)], validation_split=0.1)
        return cvae_hist

    def vae_loss(self, y_true, y_pred):
        # E[log P(X|z)]
        recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)
        # D_KL(Q(z|X) || P(z|X))
        kl = 0.5 * K.sum(K.exp(self.sigma) + K.square(self.mu) - 1. - self.sigma, axis=-1)
        return K.mean(recon + kl)

    def kl_loss(self):
        return 0.5 * K.sum(K.exp(self.sigma) + K.square(self.mu) - 1. - self.sigma, axis=-1)

    def recon_loss(self, y_true, y_pred):
        return K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1)

    def plot_latent_space(self, x, y, name):
        x = np.asarray(x.values).astype(np.float32)
        y = np.asarray([np.asarray(ele) for ele in y]).astype(np.float32)
        encoded = self.encoder.predict([x, y])[2]
        plt.figure(figsize=(6, 6))
        plt.scatter(encoded[:, 0], encoded[:, 1], c=y[:, 0])
        plt.colorbar()
        plt.savefig('results/{}_latent_space.jpg'.format(name))
        plt.show()


    def generate_samples(self):
        z_sample = np.random.rand(1, 2)
        c = np_utils.to_categorical(0, 2)
        x_decoded = self.decoder.predict(K.concatenate([z_sample, c]))
        print(x_decoded)

    def plot_training(self):
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        plt.plot(self.c_vae.history.history['loss'])
        ax1.set_title('loss')
        ax2 = fig.add_subplot(1, 2, 2)
        plt.plot(self.c_vae.history.history['val_loss'])
        ax2.set_title('validation loss')
        plt.show()