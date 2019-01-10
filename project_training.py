from PIL import Image
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
import numpy as np
import scipy.misc
import scipy
import imageio
import sys
import torch
import resnet
import torchvision
import math
import os
import pickle

feature_vector_array = pickle.Unpickler( open("test.txt", 'rb') ).load()
feature_vector_array = np.float32(feature_vector_array)
feature_vector_array = feature_vector_array[1:]
print( np.size(feature_vector_array,0) )
print(feature_vector_array)

class_sizes = [0,40,40,35,45,40,40,39,40,39,40]

y = np.zeros((1,1), float)
n = 0
for m in class_sizes:
    classifier = np.zeros((m,1), float) + n
    y = np.vstack((y, classifier))
    n = n + 1
    
y = y[1:]
X_train, X_test, y_train, y_test = train_test_split(feature_vector_array, y, test_size = 0.2)

mdl = svm.SVC(kernel = 'rbf', gamma = 0.5, C = 500.0, max_iter = -1)
mdl.fit(X_train, y_train)

pickle.dump(mdl, open('model.pickle', 'wb'))

y_pred = mdl.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (mdl, metrics.classification_report(y_test, y_pred)))
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
mdl.get_params()
