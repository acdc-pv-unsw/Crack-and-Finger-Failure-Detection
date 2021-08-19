import random
#from collections import OrderedDict
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import scipy.stats as stat
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
import scipy.io as sio
import imageio
import glob
from os import listdir
from os.path import isfile, join
from sklearn.svm import SVC
from sklearn import svm
from sklearn import metrics
import seaborn as sbn
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import cm
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import utility as utils
from sklearn.model_selection import PredefinedSplit

class TrainClassifiers:

    def __init__(self, x_train, y_train, x_test=None, y_test=None, x_val=None, y_val=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = x_val
        self.y_val = y_val

    def svm_class(self):
        clf = svm.SVC()
        #clf.fit(self.x_train, self.y_train)
        #Y_pred = clf.predict(self.x_test)

        title = "SVM statistical parameters"
        print("=" * np.max([0, np.int((40 - len(title)) / 2) + (40 - len(title)) % 2]) + " " + title + " " + "=" *
              np.max([0, np.int((40 - len(title)) / 2)]))

        x, y, ps = self.validation_split()

        # Use grid search with the validation set.
        clf_grid = self.svm_grid_search(clf, x, y, ps)

        # Predict with the test set after SVM has been trained with test and validation
        Y_pred = clf_grid.predict(self.x_test)

        self.print_stats(self.y_test, Y_pred)
        self.make_heatmap(self.y_test, Y_pred, title)

    def svm_grid_search(self, svm, x, y, ps):
        # Set the parameters by cross-validation
        tuned_parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 1e-5],
                             'C': [1, 10, 100, 1000, 5000]}


        print("# Tuning hyper-parameters")
        print()

        clf = GridSearchCV(svm, tuned_parameters, cv=ps, scoring='f1_macro', verbose=3, n_jobs=-1)
        clf.fit(x, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y, clf.predict(x)
        print(classification_report(y_true, y_pred))
        print()

        return clf

    def rf_class(self):
        # Hyper parameters from the article. (Number of trees: 5-10, min number of splits: 25, max depth: 5.
        rf = RandomForestClassifier()
        #rf.fit(self.x_train, self.y_train)
        #Y_pred = rf.predict(self.x_test)

        x, y, ps = self.validation_split()

        # Use grid search with the validation set.
        rf_grid = self.rf_grid_search(rf, x, y, ps)

        Y_pred = rf_grid.predict(self.x_test)

        title = "RF statistical parameters"
        print("="*np.max([0,np.int((40-len(title))/2)+(40-len(title))%2])+" "+title+" "+"="*np.max([0,np.int((40-len(title))/2)]))

        self.print_stats(self.y_test, Y_pred)
        self.make_heatmap(self.y_test, Y_pred, title)

    def rf_grid_search(self, rf, x, y, ps):
        # Set the parameters by cross-validation
        tuned_parameters = {'n_estimators': [5, 10], 'max_depth': [5],
                            'min_samples_split': [25]}

        print("# Tuning hyper-parameters")
        print()

        clf = GridSearchCV(rf, tuned_parameters, cv=ps, scoring='f1_macro', refit=True, verbose=0)
        clf.fit(x, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_val, clf.predict(x_val)
        print(classification_report(y_true, y_pred))
        print()

        return clf

    def knn_class(self):
        knn = KNeighborsClassifier()  # metric='minkowski', weights='distance'
        #knn.fit(self.x_train, self.y_train)
        #Y_pred = knn.predict(X_test)

        x, y, ps = self.validation_split()

        title = "k-NN statistical parameters"
        print("=" * np.max([0, np.int((40 - len(title)) / 2) + (40 - len(title)) % 2]) + " " + title +
              " " + "=" * np.max([0, np.int((40 - len(title)) / 2)]))

        # Use grid search with the validation set.
        knn_grid = self.knn_grid_search(knn, x, y, ps)

        # Predict with the test set after k-NN has been trained with test and validation
        Y_pred = knn_grid.predict(self.x_test)

        self.print_stats(self.y_test, Y_pred)
        self.make_heatmap(self.y_test, Y_pred, title)

    def knn_grid_search(self, knn, x, y, ps):

        tuned_parameters = {'n_neighbors': [1, 2, 3, 4, 5, 10, 100, 150]}

        print("# Tuning hyper-parameters")
        print()

        clf = GridSearchCV(knn, tuned_parameters, cv=ps, verbose=3)
        clf.fit(x, y)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print(clf.best_estimator_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_val, clf.predict(x_val)
        print(classification_report(y_true, y_pred))
        print()

        return clf

    def make_heatmap(self, Y_true, Y_pred, title):
        df = pd.DataFrame()
        df['Y_actual'] = Y_true
        df['Y_pred'] = Y_pred

        cm = pd.crosstab(df['Y_actual'], df['Y_pred'], rownames=['Actual'], colnames=['Predicted'])
        sbn.heatmap(cm, cmap="Blues", annot=True, fmt="d", robust=True)

        plt.close()
        plt.figure(figsize=(10, 10))
        ax1 = plt.gca()
        sbn.heatmap(
            cm,
            annot=True,
            ax=ax1,
            cmap=plt.cm.get_cmap('viridis'),
            fmt="d",
            square=True,
            linewidths=.5,
            linecolor='k',
            # vmin=0.9,
            # vmax=1+np.power(10,(1+int(np.log10(len(Y_test))))),
            # norm=mpl.colors.LogNorm(vmin=0.9,vmax=1+np.power(10,(1+int(np.log10(len(Y_test)))))),
        )
        ax1.set_xlabel('Predicted labels', fontsize=14)
        ax1.set_ylabel('Actual labels', fontsize=14)
        ax1.grid(which='minor', linewidth=0)
        ax1.grid(which='major', linewidth=0)
        plt.minorticks_on()
        plt.title(label=title, fontsize=25)
        plt.show()
        plt.close()

    def print_stats(self, Y_true, Y_pred):
        print("Acc:", metrics.accuracy_score(Y_true, Y_pred))
        print("Recall:", metrics.recall_score(Y_true, Y_pred, average='macro'))
        print("Precision:", metrics.precision_score(Y_true, Y_pred, average='macro'))
        f1score = f1_score(Y_true, Y_pred, average='macro')
        print("f1:", f1score)

    def train_all(self):
        self.svm_class()
        self.rf_class()
        self.knn_class()

    def validation_split(self):
        train_indices = np.full((len(self.x_train),), -1, dtype=int)
        val_indices = np.full((len(self.x_val),), 0, dtype=int)
        test_fold = np.append(train_indices, val_indices)
        ps = PredefinedSplit(test_fold)
        temp_x_train = self.x_train
        temp_y_train = self.y_train
        x = temp_x_train.append(self.x_val)
        y = np.append(temp_y_train, self.y_val)
        return x, y, ps

    def split_dataset(self, x_train, y_train, train_ratio, test_ratio, val_ratio):
        X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train,
                                                            test_size=1 - train_ratio, shuffle=True)

        X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test,
                                                        test_size=test_ratio / (test_ratio + val_ratio))

        return X_train, X_test, Y_train, Y_test, X_val, Y_val
