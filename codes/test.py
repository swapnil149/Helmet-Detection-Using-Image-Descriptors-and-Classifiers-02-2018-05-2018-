# -*- coding: utf-8 -*-
"""
Created on Wed May 09 08:26:04 2018

@author: DELL API
"""


import os
import cv2
import pickle
from skimage.feature import hog
import numpy as np
from vehicle_tracking_2.helpers import convert 
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.misc import imread
from sklearn.externals import joblib

#path_features = 'E:/hog_dict_skim.pkl'
#filename = 'E:/class_svm_skim.pkl'
#sc_fil = 'E:/scaler_tr.pkl'
#unsc_fil='E:/unscaled.pkl'
#path_txt = 'E:/asdas.txt'
#hog_feat_dict = pickle.load(open(path_features,'rb'))
#
#lis_veh_imgs = [i + ".png" for i in open(path_txt,'r').read().split("\n")]
#hog_feat_dict = pickle.load(open(path_features,'rb'))
#lis_nonveh_imgs = set(hog_feat_dict.keys()) - set(lis_veh_imgs)
#
#vehicle_features = np.asarray([hog_feat_dict[l] for l in lis_veh_imgs])
#
#non_vehicle_features = np.asarray([hog_feat_dict[k] for k in lis_nonveh_imgs])
#
#total_vehicles,total_nonvehicles = vehicle_features.shape[0],non_vehicle_features.shape[0]
#print total_vehicles
#print total_nonvehicles
#
#unscaled_x=joblib.load(unsc_fil)
#scaler = joblib.load(sc_fil)

path_txt = 'E:/helmet.txt'
path_features = 'E:/hog_dict_helmet.pkl'
unsc_fil_hel='E:/unscaled_hel.pkl'
unscaled_x1=joblib.load(unsc_fil_hel)
sc_fil_hel='E:/scaled_hel.pkl'

lis_helmet_imgs = [i + ".png" for i in open(path_txt,'r').read().split("\n")]
hog_feat_dict = pickle.load(open(path_features,'rb'))
lis_nonhel_imgs = set(hog_feat_dict.keys()) - set(lis_helmet_imgs)

helmet_features = np.asarray([hog_feat_dict[l] for l in lis_helmet_imgs])
non_helmet_features = np.asarray([hog_feat_dict[k] for k in lis_nonhel_imgs])


total_helmet,total_nonhelmet = helmet_features.shape[0],non_helmet_features.shape[0]
# Preprocessing features
print total_helmet
print total_nonhelmet
unscaled_x = np.vstack((helmet_features, non_helmet_features)).astype(np.float64)
scaler = StandardScaler().fit(unscaled_x)

filename='E:/class_SVM_hel_sc.pkl'
path='E:/test_hel_images/'
name='6767_3.png'
helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)
#helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)
helmet_features = hog(helmet,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)

x = scaler.transform(np.expand_dims(helmet_features,axis=0))
y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))

t_start = time.time()
svc_new = joblib.load(filename)
pre=svc_new.predict(x)
print pre
#x = scaler.transform(unscaled_x)
#y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))
#
#
#t_start = time.time()
#svc_new = joblib.load(filename)
#pre=svc_new.predict(x)
#print pre
#path='E:/test_images/'
#name='count_1.png'
#vehicle= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)
#vehicle_image = cv2.resize(vehicle, (120,200), interpolation = cv2.INTER_CUBIC)
#vehicle_features = hog(vehicle_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)
#
#x = scaler.transform(np.expand_dims(vehicle_features,axis=0))
#y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles)))
#
#t_start = time.time()
#svc_new = joblib.load(filename)
#pre=svc_new.predict(x)
#print pre

path_new='E:/test_images/'
img_new=cv2.imread(os.path.join(path_new,name),cv2.IMREAD_GRAYSCALE)
image1=np.stack((img_new,)*3,-1)
font = cv2.FONT_HERSHEY_SIMPLEX
if pre[0]==1.0:
    new_img=cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('image',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    new_img=cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('image',new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
