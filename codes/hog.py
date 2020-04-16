# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:57:32 2018

@author: DELL API
"""

import os
import cv2
import pickle

#list = os.listdir('E:/images_segment_new_0.8_2/')  
##dir is your directory path
#number_files = len(list)
##print number_files

from skimage.feature import hog
import numpy as np
from vehicle_tracking_2.helpers import convert 

class FeatureSourcer:
  def __init__(self, p, start_frame):
    
    self.color_model = p['color_model']
    self.s = p['bounding_box_size']
    
    self.ori = p['number_of_orientations']
    self.ppc = (p['pixels_per_cell'], p['pixels_per_cell'])
    self.cpb = (p['cells_per_block'], p['cells_per_block']) 
    self.do_sqrt = p['do_transform_sqrt']

    self.ABC_img = None
    self.dims = (None, None, None)
    self.hogA, self.hogB, self.HogC = None, None, None
    self.hogA_img, self.hogB_img, self.hogC = None, None, None
    
    self.RGB_img = start_frame
    self.new_frame(self.RGB_img)

  def hog(self, channel):
    features, hog_img = hog(channel, 
                            orientations = self.ori, 
                            pixels_per_cell = self.ppc,
                            cells_per_block = self.cpb, 
                            transform_sqrt = self.do_sqrt, 
                            visualise = True, 
                            feature_vector = False)
    return features, hog_img

  def new_frame(self, frame):
    
    self.RGB_img = frame 
    self.ABC_img = convert(frame, src_model= 'rgb', dest_model = self.color_model)
    self.dims = self.RGB_img.shape
    
    self.hogA, self.hogA_img = self.hog(self.ABC_img[:, :, 0])
    self.hogB, self.hogB_img = self.hog(self.ABC_img[:, :, 1])
    self.hogC, self.hogC_img = self.hog(self.ABC_img[:, :, 2])
    
  def slice(self, x_pix, y_pix, w_pix = None, h_pix = None):
        
    x_start, x_end, y_start, y_end = self.pix_to_hog(x_pix, y_pix, h_pix, w_pix)
    
    hogA = self.hogA[y_start: y_end, x_start: x_end].ravel()
    hogB = self.hogB[y_start: y_end, x_start: x_end].ravel()
    hogC = self.hogC[y_start: y_end, x_start: x_end].ravel()
    hog = np.hstack((hogA, hogB, hogC))

    return hog 

  def features(self, frame):
    self.new_frame(frame)
    return self.slice(0, 0, frame.shape[1], frame.shape[0])

  def visualize(self):
    return self.RGB_img, self.hogA_img, self.hogB_img, self.hogC_img

  def pix_to_hog(self, x_pix, y_pix, h_pix, w_pix):

    if h_pix is None and w_pix is None: 
      h_pix, w_pix = self.s, self.s
    
    h = h_pix // self.ppc[0]
    w = w_pix // self.ppc[0]
    y_start = y_pix // self.ppc[0]
    x_start = x_pix // self.ppc[0]
    y_end = y_start + h - 1
    x_end = x_start + w - 1
    
    return x_start, x_end, y_start, y_end

from skimage.feature import hog
import cv2


path='E:/images_segment_new_0.8_2/'
myimages = [] #list of image filenames
dirFiles = os.listdir(path)
dirFiles.sort(key=lambda f: int(filter(str.isdigit, f)))
for files in dirFiles: #filter out all non jpgs
    myimages.append(files)
#print len(myimages)
#print myimage

feature_params = {
  'color_model': 'yuv',                # hls, hsv, yuv, ycrcb
  'bounding_box_size': 64,             # 64 pixels x 64 pixel image
  'number_of_orientations': 9,        # 6 - 12
  'pixels_per_cell': 8,                # 8, 16
  'cells_per_block': 2,                # 1, 2
  'do_transform_sqrt': True
}
#
#with open("E:/hog_dict.pkl",'rb') as f:
#    new_dic = pickle.load(f)
#    
count=0
dic={}
print(len(myimages))
while count<len(myimages):
#while count<3000:
    #vehicle_image= cv2.imread(os.path.join(path,'8274_1'+'.png'))
    vehicle= cv2.imread(os.path.join(path,myimages[count]),cv2.IMREAD_GRAYSCALE)
    vehicle_image = cv2.resize(vehicle, (120,200), interpolation = cv2.INTER_CUBIC)
#    print(vehicle_image.shape)
#    source = FeatureSourcer(feature_params, vehicle_image)
#    vehicle_features = source.features(vehicle_image)
    vehicle_features = hog(vehicle_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)
    dic[myimages[count]]=vehicle_features
    count+=1
    if count%1000==0:
        print(count)
#print dic

with open("E:/hog_dict_skim.pkl",'wb') as f:
    pickle.dump(dic, f, protocol=pickle.HIGHEST_PROTOCOL)

    #rgb_img, y_img, u_img, v_img = source.visualize()
    
#    import matplotlib.pyplot as plt
#    #%matplotlib inline
#    plt.rcParams['figure.figsize'] = [8,8]
#    
#    print vehicle_features
#    plt.imshow(y_img)