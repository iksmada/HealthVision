# -*- coding: utf-8 -*-

print('starting code...')
import os
import sys
import time
print('loading stuff...')
import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pickle
print('loading more stuff...')
from sklearn.model_selection import KFold, GridSearchCV
from imblearn.over_sampling import SMOTE, RandomOverSampler
import datetime
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.metrics import roc_auc_score
from sklearn_porter import Porter

###############################################################################
# read data file
print('Loading data...')
numpy_vars = {}
data = np.loadtxt('train_final')
test_data = np.loadtxt('test_final')

x = data[:,:data.shape[1]-1].astype('float32')
y = data[:,-1].astype(int)

x_val = test_data[:,:data.shape[1]-1].astype('float32')
y_val = test_data[:,-1].astype(int)

###############################################################################


#Run Best model with Augmentation
print ("running best model... ")
data=data
test_data=test_data

x = data[:,:data.shape[1]-1].astype('float32')
y = data[:,-1].astype(int)
 
x_val = test_data[:,:data.shape[1]-1].astype('float32')
y_val = test_data[:,-1].astype(int)


print('smoting...')
sm = SMOTE(kind='svm')
X_res, y_res = sm.fit_sample(x, y)

clf = RandomForestClassifier(n_estimators=10, criterion='entropy', bootstrap=True, n_jobs=-1, random_state=1, verbose=1, max_depth=10)

rmse_test = []
acc = []
rmse_md = []
acc_md = []

X_train, X_test = X_res, X_res
y_train, y_test = y_res, y_res

_time = time.time()
start_time = datetime.datetime.now()
print (start_time)
clf = clf.fit(X_train, y_train)  # training phase
print("Training set score: %f" % clf.score(X_train, y_train))
print("Test set score: %f" % clf.score(X_test, y_test))

print("--- [FIT] %s seconds ---" % (time.time() - _time))

start_time = datetime.datetime.now()
print (start_time)
y_pred = clf.predict(x_val)
print("--- [Predict] %s seconds ---" % (time.time() - _time))

b = (y_pred == y_val)
rmse_md.append( np.sqrt( np.mean(np.square(y_val-y_pred)) ) )
rmse = ( np.sqrt( np.mean(np.square(y_val-y_pred)) ) )
acc_md.append( float(sum(b))/len(b)  )
cont_fold = 1

print ("Accuracy:",float(sum(b))/len(b))
print ("RMSE_test:",rmse)
rmse_test.append( np.mean(rmse_md) )
acc.append( np.mean(acc_md) )
start_time = datetime.datetime.now()
print (start_time)

print ("#######################################################")
print ("Final ACC-mean: ",np.mean(acc))
print ("Final RMSE-mean: ",np.mean(rmse_test))
print ("F1 Score:", f1_score(y_val, y_pred, average='macro'))
print ("#######################################################")
print ("saving model... ")
pickle.dump(clf, open('model_tree.pickle', 'wb'))
print ("loading model... ")
clf = pickle.load(open('model_tree.pickle', 'rb'))
print ("#######################################################")
print ("testing model... ")
result = clf.score(x_val, y_val)
print("ACC Valadation:",result)

porter = Porter(clf, language='java')
output = porter.export(embed_data=True)
f1 = open("output_porter", "wt")
f1.write(output + "\n")

result = clf.score(x_val, y_val)
print("ACC Valadation:",result)

start_time = datetime.datetime.now()
print (start_time)


 
n_classes=2
class_names = class_names=map(str, range(n_classes))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(n_classes)
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_val, y_pred)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
