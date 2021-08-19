for i in range(1):
    import pandas as pd
    import numpy as np
    import os
    import time
    import matplotlib.pyplot as plt
    import scipy.stats as stat
    from scipy.io import loadmat
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    import pickle as pkl
    import cv2
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
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from matplotlib import cm
    from sklearn.feature_selection import SelectFromModel
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier


def create_label(folder):
    #foldername = r'C:\Users\frede\Desktop\the_true_dataset\Scenario2\Train\%s' % folder
    foldername =r'C:\Users\hrp\Aalborg Universitet\DronEL AAU - Documents\RESOURCES\ML paper code\%s' % folder
    filelist = os.listdir(foldername)
    Labels = []
    for file in filelist:
        if folder == 'fingerfailuremask':
            Labels.append(1)
        elif folder == 'crackcmask':
            Labels.append(0)
        elif folder == 'healthymask':
            Labels.append(2)
    return Labels


def extract_images(folder):
    #foldername = r'C:\Users\frede\Desktop\the_true_dataset\Scenario2\Train\%s' % folder
    foldername = r'C:\Users\hrp\Aalborg Universitet\DronEL AAU - Documents\RESOURCES\ML paper code\%s' % folder
    onlyfiles = [f for f in listdir(foldername) if isfile(join(foldername, f))]
    # Python way
    img_list = [cv2.resize(cv2.imread(join(foldername, file), cv2.IMREAD_GRAYSCALE), (200, 200),
                           interpolation=cv2.INTER_NEAREST) for file in onlyfiles]
    img_list_flat = [img.flatten('F') for img in img_list]
    return img_list_flat

def build_failures(failures):
    enc = OneHotEncoder()
    failures_labels = failures['labels'].values
    failures_labels = np.reshape(failures_labels, (-1, 1))
    failures_labels = enc.fit_transform(failures_labels).toarray().tolist()
    result = failures.drop(['labels'], axis=1)
    result['labels'] = failures_labels

    return result

Labels0 = create_label('crackcmask')
Labels1 = create_label('fingerfailuremask')
Labels2 = create_label('healthymask')
Labels = Labels0 + Labels1 + Labels2
images0 = extract_images('crackccells')
images1 = extract_images('fingerfailurecells')
images2 = extract_images('healthycells')
concatImages = images0 + images1 + images2
len(concatImages)
len(Labels)

unique=[np.unique(img) for img in concatImages]
ELhistList = [np.bincount(img) for img in concatImages]
ELhistList = [np.concatenate((hist,np.array([0]*(256-len(hist))))) for hist in ELhistList]

plt.figure(figsize=(10,10))
ax=plt.axes()
for img in concatImages:
    sbn.distplot(img,bins=256,ax=ax)
    ax.set_xlim(xmin=0,xmax=255)
    plt.plot(ELhistList)

imgTest = concatImages[18]

np.unique(imgTest)
ELhist=np.bincount(imgTest)
len(ELhist)
plt.plot(ELhist)
imgTest = img_list_flat[22]

np.unique(imgTest)
ELhist=np.bincount(imgTest)
len(ELhist)
plt.plot(ELhist)

stat_feature = {'mu':[],'ICA':[],'kur':[],'skew':[], 'sp':[], 'md':[], 'sd':[], 'var':[],
                '25p':[], '75p':[],'fw':[],'kstat':[],'entropy':[]}

for ELhist in ELhistList:
    PEL=ELhist/np.sum(ELhist)
    stat_feature['mu'].append(np.mean(PEL))
    stat_feature['md'].append(np.median(PEL))
    threshold=min(25,len(PEL)-1)
    stat_feature['ICA'].append(100*np.sum([PEL[i] for i in range(threshold+1)]))
    stat_feature['sd'].append(np.std(PEL))
    stat_feature['kur'].append(stat.kurtosis(PEL))
    stat_feature['skew'].append(stat.skew(PEL))
    stat_feature['var'].append(stat.variation(PEL))
    stat_feature['sp'].append(np.ptp(PEL))
    stat_feature['fw'].append(((5/100)*(np.max(PEL)))-((5/100)*(np.min(PEL))))
    stat_feature['25p'].append(np.quantile(PEL, 0.25, interpolation='lower'))
    stat_feature['75p'].append(np.quantile(PEL, 0.75, interpolation='higher'))
    stat_feature['entropy'].append(stat.entropy(PEL))
    stat_feature['kstat'].append(stat.kstat(PEL))

from sklearn.preprocessing import MinMaxScaler
dfhist = pd.DataFrame(stat_feature)
scaler = MinMaxScaler()
normdfhist= pd.DataFrame(scaler.fit_transform(dfhist))
X=pd.DataFrame(ELhistList)
Xfinal=pd.merge(X,dfhist,left_index=True,right_index=True)
sortdf = pd.DataFrame(Xfinal)
sortdf['labels']=Labels
Y=Labels
len (Y)
X=sortdf
len (X)
#sortdf = build_failures(sortdf)
sortdf.to_pickle('training_2.pkl')
print('FILE CREATED')
#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---------------- SVM Algorithm implementation ----------------
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
for i in range(1):
    X=sortdf[sortdf.columns[256:268]].values
    Y=Labels
    train_ratio = 0.60
    validation_ratio = 0.20
    test_ratio = 0.20
    # train is now 60% of the entire data set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1 - train_ratio)
    # test is now 20% of the initial data set
    # validation is now 20% of the initial data set
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    print(X_train, X_val, X_test, Y_val, Y_test)
    scoring = ['precision_micro', 'recall_micro']
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    X_val=sc.transform(X_val)
    clf = svm.SVC(C=2.0, kernel='rbf', decision_function_shape='ovo', tol=1e-05)
    clf.fit(X_train,Y_train)
    clf.predict(X_train)
    Y_pred = clf.predict(X_test)
    Y_pred1 = clf.predict(X_val)
    print(Y_pred)
    print(Y_pred1)
    f1=precision_recall_fscore_support(Y_test,Y_pred, average='micro')
    print(f1)
    f12=precision_recall_fscore_support(Y_val,Y_pred, average='micro')
    print(f12)
    Score = clf.score(X_test,Y_test)
    Score1 = clf.score(X_train,Y_train)
    Score2 = clf.score(X_val,Y_val)
    print(precision_recall_fscore_support(Y_test, Y_pred, average=None))
    print(precision_recall_fscore_support(Y_val, Y_pred1, average=None))
    print(classification_report(Y_test, Y_pred))
    print(classification_report(Y_val, Y_pred1))
    print(classification_report(Y_train, clf.predict(X_train)))
    f1score=f1_score(Y_test, Y_pred, average=None)
    print(f1score)
    f1score1=f1_score(Y_val, Y_pred1, average=None)
    print(f1score1)
    acc=accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
    print("Accuracy: %0.1f (+/- %0.1f)" % (Score.mean(), Score.std() * 2))
    print(acc)
    acc1=accuracy_score(Y_val, Y_pred1, normalize=True, sample_weight=None)
    print("Accuracy: %0.1f (+/- %0.1f)" % (Score.mean(), Score.std() * 2))
    print(acc1)
    df=pd.DataFrame()
    df['Y_actual']=Y_test
    df['Y_pred']=Y_pred
    df['Y_val']=Y_val
    df['Y_pred1']=Y_pred1
    CM = pd.crosstab(df['Y_actual'], df['Y_pred'], rownames=['Actual'], colnames=['Predicted'])
    sbn.heatmap(CM,cmap="Blues",annot=True, fmt="d",robust=True)
    print(CM)
    CM1 = pd.crosstab(df['Y_val'], df['Y_pred1'], rownames=['Actual'], colnames=['Predicted'])
    sbn.heatmap(CM1,cmap="YlGnBu",annot=True, fmt="d",robust=True)
    print(CM1)
    #\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#---------------- Random Forest Algorithm implementation ----------------
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
for i in range(1):
    X=sortdf[sortdf.columns[256:268]].values
    Y=Labels
    train_ratio = 0.60
    validation_ratio = 0.20
    test_ratio = 0.20
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1 - train_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    print(X_train, X_val, X_test, Y_val, Y_test)
    sc=StandardScaler()
    X_train=sc.fit_transform(X_train)
    X_test=sc.transform(X_test)
    rf = RandomForestClassifier(n_estimators=10, min_samples_split=25, criterion='gini', max_depth=5, max_features=None)
    rf.fit(X_train,Y_train)
    rf.predict(X_train)
    Y_pred = rf.predict(X_test)
    Score = rf.score(X_test,Y_test)
    Score1 = rf.score(X_train,Y_train)
    print(Score)
    df=pd.DataFrame()
    df['Y_actual']=Y_test
    df['Y_pred']=Y_pred
    df
    print(precision_recall_fscore_support(Y_test, Y_pred, average=None))
    print(classification_report(Y_test, Y_pred))
    print(classification_report(Y_train, clf.predict(X_train)))
    f1score=f1_score(Y_test, Y_pred, average=None)
    print(f1score)
    acc=accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
    print("Accuracy: %0.1f (+/- %0.1f)" % (Score.mean(), Score.std() * 2))
    CM = pd.crosstab(df['Y_actual'], df['Y_pred'], rownames=['Actual'], colnames=['Predicted'])
    sbn.heatmap(CM,cmap="Blues",annot=True, fmt="d",robust=True)
    print(CM)

#\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
#------------------------------K-Nearest Neighbors--------------------------
#///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
for i in range(1):
    X=sortdf.columns[20:254].values
    Y=Labels
    train_ratio = 0.60
    validation_ratio = 0.20
    test_ratio = 0.20
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1 - train_ratio)
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    sc=StandardScaler() #Feature scaling
    X_train=sc.fit_transform(X_train) #Feature scaling
    X_test=sc.transform(X_test) #Feature scaling
    knn=KNeighborsClassifier(n_neighbors=2, metric='minkowski' ) #weights='distance'
    knn.fit(X_train,Y_train)
    Y_pred=knn.predict(X_test)
    print(Y_pred)
    df=pd.DataFrame()
    df['Y_actual']=Y_test
    df['Y_pred']=Y_pred
    df
    Score = knn.score(X_test,Y_test)
    Score1 = knn.score(X_train,Y_train)
    print(precision_recall_fscore_support(Y_test, Y_pred, average=None))
    print(classification_report(Y_test, Y_pred))
    print(classification_report(Y_train, knn.predict(X_train)))
    f1score=f1_score(Y_test, Y_pred, average=None)
    print(f1score)
    acc=accuracy_score(Y_test, Y_pred, normalize=True, sample_weight=None)
    print("Accuracy: %0.1f (+/- %0.1f)" % (Score.mean(), Score.std() * 2))
    CM = pd.crosstab(df['Y_actual'], df['Y_pred'], rownames=['Actual'], colnames=['Predicted'])
    sbn.heatmap(CM,cmap="Blues",annot=True, fmt="d",robust=True)
    print(CM)
