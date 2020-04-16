# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 00:23:41 2018

@author: DELL API
"""

import os
#from PIL import Image
import numpy as np
import cv2
#from tqdm import tqdm
#import matplotlib.pyplot as plt
#%matplotlib inline

path_fgmask='E:/images_fmask_new/'
path_original='E:/images_gray/'
#cap=cv2.imread(os.path.join(path_fgmask,str(1)+'.png'))
#ret,frame = cap.read()
#img1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
count = 1
#cap2=cv2.imread(os.path.join(path_original,str(1)+'.png'))
#total=os.path.getsize(path_original)
#print total
#img1=cv2.imread(os.path.join(path_fgmask,str(count)+'.png'),cv2.IMREAD_GRAYSCALE)

img_change = cv2.imread(os.path.join(path_original,str(count)+'.png'),cv2.IMREAD_GRAYSCALE)
ret = True
total=0
while count<=17940:
    
    #cv2.imwrite(os.path.join(path_original,str(count)+'.png'),frame)
    #fgmask = fgbg.apply(frame)
    img1=cv2.imread(os.path.join(path_fgmask,str(count)+'.png'),cv2.IMREAD_GRAYSCALE)
    #img_change = cv2.imread(os.path.join(path_original,str(count)+'.png'),cv2.IMREAD_GRAYSCALE)
    
    img_test = img_change.copy()
    blur = cv2.GaussianBlur(img1,(5,5),0.8)
    ret3,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel2 = np.ones((2,2),np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2)
    _,contours, hier = cv2.findContours(closing, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2
    path_segment='E:/images_segment/'
    if not os.path.exists(path_segment):
        os.makedirs(path_segment)
    idx=0
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv2.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        
        #cv2.putText(img_test,"Area : "+str(h*w),(x+w,h+w),fontScale,fontColor,lineType)
        #font = cv2.FONT_HERSHEY_SIMPLEX
        if h*w > 4000 and h*w < 40000:
            cv2.rectangle(img_test, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #cv2.putText(img_test,'Area: '+str(h*w),(x+w,y+h), font, 0.5,(255,255,255),2,cv2.LINE_AA)
            idx+=1
            new_img=img_test[y:y+h,x:x+w]
            cv2.imwrite(os.path.join(path_segment,str(count)+"_"+str(idx)+'.png'), new_img)
            total+=1
    
    count+=1
    img_change = cv2.imread(os.path.join(path_original,str(count)+'.png'),cv2.IMREAD_GRAYSCALE)
    #ret, frame = cap.read()
    #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #count += 1
    if count%1000 == 0:
        print(count)
    
# cv2.drawContours(img_test, contours, -1, (255, 255, 0), 1)
print total