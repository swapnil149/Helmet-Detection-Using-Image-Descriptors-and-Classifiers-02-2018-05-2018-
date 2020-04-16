# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:29:04 2018

@author: DELL API
"""

import numpy as np
import cv2
"""
#img1 = cv2.cvtColor(cv2.imread('img0100.png'), cv2.COLOR_BGR2GRAY)
##cv2.imwrite('gray1.png',img1)
#img2 = cv2.cvtColor(cv2.imread('img0282.png'), cv2.COLOR_BGR2GRAY)
##cv2.imwrite('gray2.png',img2)
#img3 = cv2.cvtColor(cv2.imread('img0466.png'), cv2.COLOR_BGR2GRAY)
#cv2.imwrite('gray3.png',img3)
#img4 = cv2.cvtColor(cv2.imread('img2765.png'), cv2.COLOR_BGR2GRAY)
#cv2.imwrite('gray4.png',img4)
img1 = cv2.cvtColor(cv2.imread('img0100.png'),cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(cv2.imread('img0282.png'),cv2.COLOR_BGR2RGB)
img3 = cv2.cvtColor(cv2.imread('img0466.png'),cv2.COLOR_BGR2RGB)
img4 = cv2.cvtColor(cv2.imread('img2765.png'),cv2.COLOR_BGR2RGB)

fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#ret,frame1=img1.read()

fgmask1=fgbg.apply(img1)
cv2.imshow('frame1',fgmask1)

#ret,frame2=img2.read()
fgmask2=fgbg.apply(img2)
cv2.imshow('frame2',fgmask2)

#ret,frame3=img3.read()
fgmask3=fgbg.apply(img3)
cv2.imshow('frame3',fgmask3)

#ret,frame4=img4.read()
fgmask4=fgbg.apply(img4)
cv2.imshow('frame4',fgmask4)

cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()
img=cv2.imread('gray1.png')
[h,w,l]=np.shape(img)
fgmask=np.resize(fgmask4,(h,w,l))
#print np.shape(fgmask)
"""
"""
row,col= fgmask4.shape
mean = 0
var = 0.1
sigma = var**0.5
gauss = np.random.normal(mean,sigma,(row,col))
gauss = gauss.reshape(row,col)
noisy = fgmask4 + gauss
cv2.imshow('noise',noisy)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()
"""
#from matplotlib import pyplot as plt
#img = cv2.imread('gray1.png')
"""
blur = cv2.GaussianBlur(fgmask4,(5,5),0.9)
cv2.imshow('blur',blur)
cv2.waitKey(0)  
#median_blur= cv2.medianBlur(noisy, 3)
#median = cv2.medianBlur(noisy,5)
#cv2.imshow('median',median)
#cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()
"""
"""
plt.subplot(121),plt.imshow(noisy),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(blur),plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
"""
"""
#img = cv2.imread('j.png',0)
kernel1 = np.ones((6,6),np.uint8)
#kernel1=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
opening = cv2.morphologyEx(fgmask4, cv2.MORPH_OPEN, kernel1)
cv2.imshow('open',opening)
cv2.waitKey(0)
kernel2 = np.ones((9,9),np.uint8)
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel2)
cv2.imshow('close',closing)
cv2.waitKey(0)
erosion = cv2.erode(fgmask4,kernel1,iterations = 1)
cv2.imshow('ero_img',erosion)
cv2.waitKey(0)
img_dilation = cv2.dilate(erosion, kernel1, iterations=1)
cv2.imshow('dil',img_dilation)
cv2.waitKey(0)                 # Waits forever for user to press any key
cv2.destroyAllWindows()
"""
"""
cv2.imwrite('input.png',fgmask2)

image = cv2.imread('input.png')
"""
import os
path_original='E:/images/'
path_fgmask='E:/images_fmask/'
path_test='E:/images_test/'
if not os.path.exists(path_test):
     os.makedirs(path_test)  
count=8274
img1=cv2.imread(os.path.join(path_fgmask,str(count)+'.png'))
#img.shape
#cv2.imshow('fmask',img)
#cv2.waitKey(0)
#filtering
img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
blur1 = cv2.GaussianBlur(img,(5,5),0.5)
cv2.imwrite(os.path.join(path_test,'gaussian(8274,0.5)'+'.png'), blur1)

blur2 = cv2.GaussianBlur(img,(5,5),0.9)
cv2.imwrite(os.path.join(path_test,'gaussian(8274,0.9)'+'.png'), blur2)

# ostu thresholding
ret1,th1 = cv2.threshold(blur1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(path_test,'gaussian,thres(8274,0.5)'+'.png'), th1)

ret2,th2 = cv2.threshold(blur2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite(os.path.join(path_test,'gaussian,thres(8274,0.9)'+'.png'), th2)

#closing morphological
kernel2 = np.ones((6,6),np.uint8)
kernel3=np.ones((1,1),np.uint8)
kernel5=np.ones((8,8),np.uint8)
kernel4=np.ones((3,3),np.uint8)



closing3 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel4)
closing1 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel2)
closing2 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel3)
closing4 = cv2.morphologyEx(th1, cv2.MORPH_CLOSE, kernel5)
closing5 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel4)
closing6 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel2)
closing7 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel3)
closing8 = cv2.morphologyEx(th2, cv2.MORPH_CLOSE, kernel5)

cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.5,(6,6))'+'.png'), closing1)
cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.5,(1,1))'+'.png'), closing2)
cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.5,(3,3))'+'.png'), closing3)
cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.5,(8,8))'+'.png'), closing4)
cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.9,(6,6))'+'.png'), closing5)
cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.9,(1,1))'+'.png'), closing6)
cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.9,(3,3))'+'.png'), closing7)
cv2.imwrite(os.path.join(path_test,'gaussian,thres,morp(8274,0.9,(8,8))'+'.png'), closing8)
 
"""
image = cv2.imread('example.jpg')

edged = cv2.Canny(image, 10, 250)
cv2.imshow('Edges', edged)
cv2.waitKey(0)
 
#applying closing function 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closed", closed)
cv2.waitKey(0)
"""
#finding_contours 
#(cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
_,contours,hierarchy = cv2.findContours(closing2,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE) 
for c in contours:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	cv2.drawContours(closing2, [approx], -1, (0, 255, 0), 2)
cv2.imshow("Output", closing2)
cv2.waitKey(0)
cv2.destroyAllWindows()

