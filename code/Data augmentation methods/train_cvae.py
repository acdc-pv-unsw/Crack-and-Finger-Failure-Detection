from model.ConditionalVariationalAutoencoder import ConditionaVariationalAutoencoder as CVAE
from config import args
from utils import load_file
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utility as utils
from sklearn.model_selection import train_test_split
from keras.utils.vis_utils import plot_model
import graphviz
import pydot

tf.random.set_seed(0)

class Training:
    def __init__(self):
        x_train = load_file('data/man_dataaug.pkl')

        x_train = utils.normalize_values(x_train[x_train.columns[256:270]])
        y_train = x_train['labels']
        x_train.drop(['labels', 'mu'], axis=1, inplace=True)

        x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.25, shuffle=True)

        model = CVAE(emb_size=6,
                     hidden_states=32,
                     y_dim=len(y_train.iloc[0]),
                     x_dim=x_train.shape[1],
                     batch_size=10)

        model.build()
        model.fit([x_train, y_train], [x_test, y_test], n_epochs=1)
        model.plot_training()
        model.plot_latent_space(x_train, y_train, name='train')
        model.plot_latent_space(x_test, y_test, name='test')
        model.save()

        samples = pd.DataFrame(columns=['ICA', 'kur', 'skew', 'sp', 'md',
                                        'sd', 'var', '25p', '75p', 'fw', 'kstat', 'entropy', 'labels'])
        for i in range(0, 40000):
            data = model.generate_samples([1., 0.])
            samples = samples.append(data, ignore_index=True)
            data = model.generate_samples([0., 1.])
            samples = samples.append(data)

        samples.to_pickle('data/vae_samples.pkl')
        print('FILE CREATED')


if __name__ == "__main__":
    train = Training()


