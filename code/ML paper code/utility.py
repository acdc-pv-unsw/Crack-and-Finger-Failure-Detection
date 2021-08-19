from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize_values(failures):
    failures_labels = failures['labels']
    result = failures.drop(['labels'], axis=1).apply(scale)
    result['labels'] = failures_labels

    return result


def scale(x):
    enc = MinMaxScaler()
    values = enc.fit_transform(np.reshape(x.values, (-1, 1)))[:, 0]
    return values

def plot_norm_stats(normdfhist, fig_name, type):
    scaler = MinMaxScaler()
    if type == 'vae':
        normdfhist.columns = ['ICA', 'kur', 'skew', 'sp', 'md','sd', 'var', '25p',
                              '75p', 'fw', 'kstat', 'entropy', 'labels']
    elif type == 'normal':
        normdfhist = pd.DataFrame(scaler.fit_transform(normdfhist))
        normdfhist.columns = ['mu','ICA', 'kur', 'skew', 'sp', 'md', 'sd', 'var', '25p',
                              '75p', 'fw', 'kstat', 'entropy', 'labels']
        normdfhist = normdfhist.drop(['mu'], axis=1)
    normdfhist = normdfhist.drop(['labels'], axis=1)

    plt.close()
    plt.figure(figsize=(7, 7))
    ax1 = plt.gca()
    ax1.set_xlabel('Feature vectors', fontsize=14)
    ax1.set_ylabel('Normalized values', fontsize=14)
    ax1.grid(which='minor', linewidth=0)
    ax1.grid(which='major', linewidth=0)
    plt.minorticks_on()

    x = normdfhist.mean().plot(kind='bar', width=1, ylim=(0,1))
    fig = x.get_figure()
    fig.savefig(fig_name + '.pdf')

def make_histogram(dataset):
    grouped = dataset.groupby(dataset.labels)
    class_zero = grouped.get_group(0)  # class 0 = Crack B and C
    class_one = grouped.get_group(1)  # class 1 = Crack A and finger failure
    last_columns = dataset[dataset.columns[256:270]]
    if not last_columns.empty:
        #class_two = grouped.get_group(2)  # class 2 = healthy
        plot_norm_stats(class_zero[class_zero.columns[256:270]], 'stats0', 'normal')
        plot_norm_stats(class_one[class_one.columns[256:270]], 'stats1', 'normal')
        #plot_norm_stats(class_two[class_two.columns[256:270]], 'stats2', 'normal')
    else:
        plot_norm_stats(class_zero, 'stats0_vae', 'vae')
        plot_norm_stats(class_one, 'stats1_vae', 'vae')

def make_dataset(scenario):
    train = pd.read_pickle('data/training_%s.pkl' % scenario)
    train = normalize_values(train[train.columns[256:270]])
    # Get histogram of Class 2 Healthy
    grouped = train.groupby(train.labels)
    class_two = grouped.get_group(2)  # class 2 = healthy
    plot_norm_stats(class_two, 'stats2_train', 'normal')
    y_train = train['labels'].values
    x_train = train.drop(['labels', 'mu'], axis=1)  # sets x to not contain label (and mu because mu contains bad info)

    test = pd.read_pickle('data/testing_%s.pkl' % scenario)
    test = normalize_values(test[test.columns[256:270]])
    grouped = test.groupby(test.labels)
    class_two = grouped.get_group(2)  # class 2 = healthy
    plot_norm_stats(class_two, 'stats2_test', 'normal')
    y_test = test['labels'].values  # Sets y to only contain labels
    x_test = test.drop(['labels', 'mu'], axis=1)  # sets x to not contain label (and mu because mu contains bad info)

    validation = pd.read_pickle('data/validation_%s.pkl' % scenario)
    validation = normalize_values(validation[validation.columns[256:270]])
    grouped = validation.groupby(validation.labels)
    class_two = grouped.get_group(2)  # class 2 = healthy
    plot_norm_stats(class_two, 'stats2_val', 'normal')
    y_val = validation['labels'].values  # Sets y to only contain labels
    x_val = validation.drop(['labels', 'mu'],
                            axis=1)  # sets x to not contain label (and mu because mu contains bad info)
    return x_train, y_train, x_test, y_test, x_val, y_val

# Used to convert onehot encoded label into a single int.
def make_train_test_set(fails, healthy):
    for i, row in fails.iterrows():
        if fails.at[i, 'labels'] == [1., 0.]: # Crack B and C
            fails.at[i, 'labels'] = 0
        else:
            fails.at[i, 'labels'] = 1

    fails = normalize_values(fails[fails.columns[256:270]])
    labels = fails['labels'].astype(dtype='int64')
    fails['labels'] = labels
    #fails.drop(['labels', 'mu'], axis=1, inplace=True)

    healthy = healthy.drop(['labels'], axis=1)
    healthy['labels'] = healthy['new-labels'].astype(dtype='int64')
    full_data = healthy.drop(['new-labels'], axis=1)
    grouped = full_data.groupby(full_data.labels)
    class_two = grouped.get_group(2)  # class 2 = healthy
    class_two = normalize_values(class_two[class_two.columns[256:270]]).sample(1200)
    return fails.append(class_two)
