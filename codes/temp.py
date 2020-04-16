
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

cap = cv2.VideoCapture('C:/Users/DELL API/.spyder/video1.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()
ret,frame = cap.read()
# frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
count = 1
path_fmask = 'E:/images_fmask_new/'
if not os.path.exists(path_fmask):
    os.makedirs(path_fmask)
ret = True
while ret:
    
    
    fgmask = fgbg.apply(frame)
    fgmask[fgmask==127]=0
#     cv2.imwrite(os.path.join(path_fmask,str(count)+'.png'), fgmask)
    ret, frame = cap.read()
#     frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    count += 1
    if count%1000 == 0:
        print(count)
    if count==2354:
        im_mask_test = fgmask
        break

#path_fgmask='E:/images_fmask_gray/'
path_original='E:/images_gray/'
#path_gauss='E:/images_gauss/'
#path_th='E:/images_thres/'
#path_close='E:/images_close/'

path_gauss='E:/images_gauss_0.5/'
path_th='E:/images_thres_0.5/'
path_close='E:/images_close_0.5/'
cap = cv2.VideoCapture('video1.avi')
#length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#print(length)
fgbg = cv2.createBackgroundSubtractorMOG2()
#fgbg.setDetectShadows(False)
if not os.path.exists(path_fmask):
    os.makedirs(path_fmask)
    #os.makedirs(path_original)
if not os.path.exists(path_gauss):
    os.makedirs(path_gauss)
#lis = os.listdir(path_fgmsk)
if not os.path.exists(path_original):
     os.makedirs(path_original)
if not os.path.exists(path_th):
     os.makedirs(path_th)
if not os.path.exists(path_close):
     os.makedirs(path_close)

ret,frame = cap.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
count = 1

ret = True
while ret:
    
    cv2.imwrite(os.path.join(path_original,str(count)+'.png'),frame)
    fgmask = fgbg.apply(frame)
 #   im = Image.fromarray(fgmask)
#    im.save(os.path.join(path,str(count)+'.png'))
    #cv2.imwrite(os.path.join(path_fgmask,str(count)+'.png'), fgmask)
    cv2.imwrite(os.path.join(path_fmask,str(count)+'.png'), fgmask)
    
    #gaussian filtering    
    
    #blur = cv2.GaussianBlur(fgmask,(5,5),0.5)
    blur = cv2.GaussianBlur(fgmask,(5,5),0.8)
    cv2.imwrite(os.path.join(path_gauss,str(count)+'.png'), blur)
    # ostu thresholding
    ret3,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(path_th,str(count)+'.png'), th)
    #closing morphological
    #kernel2 = np.ones((1,1),np.uint8)
    kernel2 = np.ones((2,2),np.uint8)
    closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel2)
    cv2.imwrite(os.path.join(path_close,str(count)+'.png'), closing)
    
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    count += 1
    if count%1000 == 0:
        print(count)
        
        
#print("Count ="+str(count))
cap.release()
cv2.destroyAllWindows()

