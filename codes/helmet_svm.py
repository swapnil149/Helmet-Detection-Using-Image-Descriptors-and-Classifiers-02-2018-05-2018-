# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 04:38:07 2018

@author: DELL API
"""

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.misc import imread
from sklearn.externals import joblib

import random as rand
import numpy as np 
#import cv2
#import glob
import time
import pickle

# Loading features in lists
path_txt = 'E:/helmet.txt'
path_features = 'E:/hog_dict_helmet.pkl'
lis_helmet_imgs = [i + ".png" for i in open(path_txt,'r').read().split("\n")]
hog_feat_dict = pickle.load(open(path_features,'rb'))
lis_nonhel_imgs = set(hog_feat_dict.keys()) - set(lis_helmet_imgs)

helmet_features = np.asarray([hog_feat_dict[l] for l in lis_helmet_imgs])
print(helmet_features[0:10])
non_helmet_features = np.asarray([hog_feat_dict[k] for k in lis_nonhel_imgs])
print(non_helmet_features[0:10])

total_helmet,total_nonhelmet = helmet_features.shape[0],non_helmet_features.shape[0]
# Preprocessing features
print total_helmet
print total_nonhelmet
unscaled_x = np.vstack((helmet_features, non_helmet_features)).astype(np.float64)
scaler = StandardScaler().fit(unscaled_x)
#x=unscaled_x
x = scaler.transform(unscaled_x)
y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))

#print("...Done")
#print("Time Taken:", np.round(time.time() - t_start, 2))
#print(" x shape: ", x.shape, " y shape: ", y.shape)

print("Training classifier...")
t_start = time.time()

# Training 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = rand.randint(1, 100))
svc = LinearSVC()
svc.fit(x_train, y_train)
accuracy = svc.score(x_test, y_test)

print("...Done")
print("Time Taken:", np.round(time.time() - t_start, 2))
print("Accuracy: ", np.round(accuracy, 4))
