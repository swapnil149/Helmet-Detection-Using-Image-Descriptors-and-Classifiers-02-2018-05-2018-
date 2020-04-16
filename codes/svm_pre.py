# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 22:41:11 2018

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

import time
import pickle
path_features = 'E:/hog_dict_skim.pkl'
filename = 'E:/class_svm_skim.pkl'
sc_fil = 'E:/scaler_tr.pkl'
unsc_fil='E:/unscaled.pkl'
path_txt = 'E:/asdas.txt'
hog_feat_dict = pickle.load(open(path_features,'rb'))

lis_veh_imgs = [i + ".png" for i in open(path_txt,'r').read().split("\n")]
hog_feat_dict = pickle.load(open(path_features,'rb'))
lis_nonveh_imgs = set(hog_feat_dict.keys()) - set(lis_veh_imgs)

vehicle_features = np.asarray([hog_feat_dict[l] for l in lis_veh_imgs])

non_vehicle_features = np.asarray([hog_feat_dict[k] for k in lis_nonveh_imgs])

total_vehicles,total_nonvehicles = vehicle_features.shape[0],non_vehicle_features.shape[0]
print total_vehicles
print total_nonvehicles

#unscaled_x = np.vstack((vehicle_features, non_vehicle_features)).astype(np.float64)
#print unscaled_x
#unscaled_x=joblib.load(unsc_fil)
##scaler = StandardScaler().fit(unscaled_x)
#scaler = joblib.load(sc_fil)
##print scaler
#x = scaler.transform(unscaled_x)
#y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles))
#
#t_start = time.time()
#svc_new = joblib.load(filename)
#
#pre=svc_new.predict(x[970:1000])
#print pre



#
## Training 
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
#                                                    random_state = rand.randint(1, 100))
#svc = LinearSVC()
#svc.fit(x_train, y_train)
#accuracy = svc_new.score(y,pre)
#
#
#print("Time Taken:", np.round(time.time() - t_start, 2))
#print("Accuracy: ", np.round(accuracy, 4))
