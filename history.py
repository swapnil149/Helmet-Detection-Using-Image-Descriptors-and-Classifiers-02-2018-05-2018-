# *** Spyder Python Console History Log ***
cv2.putText(img_new,'YES',(10,500), font,4 ,(200,200,250),2,cv2.LINE_AA)
plt.imshow(img_new)
cv2.imshow(img_new)
cv2.imshow('image',img_new)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
img_new=image.copy()
cv2.putText(img_new,'YES',(10,500), font,4 ,(255,0,0),2,cv2.LINE_AA)
cv2.imshow('image',img_new)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
img_new=image.copy()
cv2.putText(img_new,'YES',(10,500), font,10 ,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('image',img_new)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
plt.imshow(img_new)
cv2.putText(img_new,'YES',(10,10), font,10 ,(0,0,255),2,cv2.LINE_AA)
plt.imshow(img_new)
cv2.imshow('image',img_new)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
img_new=image.copy()
cv2.putText(img_new,'YES',(50,50), font,20 ,(0,0,255),2,cv2.LINE_AA)
plt.imshow(img_new)
cv2.imshow('image',img_new)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
cv2.putText(img_new,'YES',(50,50), font,100 ,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('image',img_new)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
new_path='E:/images_test/'
new_img=cv2.imread(os.path.join(new_path,'segmented.png'),cv2.IMREAD_GRAYSCALE)
copy_img=new_img.copy()
cv2.putText(copy_img,'YES',(50,50), font,100 ,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('image',img_new)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
cv2.imshow('image',copy_img)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
copy_img=new_img.copy()
cv2.putText(copy_img,'YES',(50,50), font,0.5 ,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('image',copy_img)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
new_copy_img = np.stack((copy_img,)*3,-1)
cv2.putText(new_copy_img,'YES',(50,50), font,0.5 ,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('image',new_copy_img)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()
image1= np.stack((img_new,)*3,-1)
cv2.putText(image1,'YES',(50,50), font,0.5 ,(0,0,255),2,cv2.LINE_AA)
cv2.imshow('image',image1)cv2.waitKey(0)                 # Waits forever for user to press any keycv2.destroyAllWindows()

##---(Wed May 09 07:50:18 2018)---
runfile('C:/Users/DELL API/.spyder/svm_pre.py', wdir='C:/Users/DELL API/.spyder')
unscaled_x=joblib.load(unsc_fil)
unsc_fil='E:/unscaled.pkl/'
unscaled_x=joblib.load(unsc_fil)
unsc_fil='E:/unscaled.pkl'
unscaled_x=joblib.load(unsc_fil)
sc_fil='E:/scaler_tr.pkl'
scaler = joblib.load(sc_fil)
x = scaler.transform(unscaled_x)
y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles))
y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles)))
t_start = time.time()
svc_new = joblib.load(filename)
filename = 'E:/class_svm_skim.pkl'
svc_new = joblib.load(filename)
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')
pre=svc_new.predict(x)
pre
img_new='count_1.png'image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre==1:    cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)else:    cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    
print pre[0]
img_new='count_1.png'image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)else:    cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    
img_new='count_1.png'image1=np.stack((img_new,)*3,-1)print image1
path='E:/test_images/'name='count_1.png'img_new=cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)else:    cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)
path='E:/test_images/'name='count_1.png'img_new=cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    new_img=cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()else:    new_img=cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()    
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')
path='E:/test_images/'name='count_3.png'vehicle= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)vehicle_image = cv2.resize(vehicle, (120,200), interpolation = cv2.INTER_CUBIC)vehicle_features = hog(vehicle_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(vehicle_features,axis=0))y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles)))t_start = time.time()svc_new = joblib.load(filename)pre=svc_new.predict(x)print pre#path='E:/test_images/'img_new=cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    new_img=cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()else:    new_img=cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()
path='E:/test_images/'name='count_4.png'vehicle= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)vehicle_image = cv2.resize(vehicle, (120,200), interpolation = cv2.INTER_CUBIC)vehicle_features = hog(vehicle_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(vehicle_features,axis=0))y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles)))t_start = time.time()svc_new = joblib.load(filename)pre=svc_new.predict(x)print pre#path='E:/test_images/'img_new=cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    new_img=cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()else:    new_img=cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')
joblib.dump(unscaled_x, '.pkl')
file_name='E:/unscaled_hel.pkl'
joblib.dump(unscaled_x, file_name)
file_name1='E:/scaled_hel.pkl'
joblib.dump(scaler, file_name)
file_name='E:/unscaled_hel.pkl'
joblib.dump(unscaled_x, file_name)
file_name1='E:/scaled_hel.pkl'
joblib.dump(scaler, file_name1)
x = scaler.transform(unscaled_x)y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))
runfile('C:/Users/DELL API/.spyder/helmet_svm.py', wdir='C:/Users/DELL API/.spyder')
filename2='E:/class_SVM_hel_sc.pkl'
joblib.dump(svc, filename2)
path_txt = 'E:/helmet.txt'path_features = 'E:/hog_dict_helmet.pkl'lis_helmet_imgs = [i + ".png" for i in open(path_txt,'r').read().split("\n")]hog_feat_dict = pickle.load(open(path_features,'rb'))lis_nonhel_imgs = set(hog_feat_dict.keys()) - set(lis_helmet_imgs)helmet_features = np.asarray([hog_feat_dict[l] for l in lis_helmet_imgs])non_helmet_features = np.asarray([hog_feat_dict[k] for k in lis_nonhel_imgs])total_helmet,total_nonhelmet = helmet_features.shape[0],non_helmet_features.shape[0]# Preprocessing featuresprint total_helmetprint total_nonhelmet
unscaled_x = np.vstack((helmet_features, non_helmet_features)).astype(np.float64)
unscaled_x
unsc_fil_hel='E:/unscaled_hel.pkl'unscaled_x1=joblib.load(unsc_fil_hel)
if unscaled_x==unscaled_x1:    print '1'
unscaled_x1
unscaled_x = np.vstack((helmet_features, non_helmet_features)).astype(np.float64)scaler = StandardScaler().fit(unscaled_x)
filename='E:/class_SVM_hel_sc.pkl'path='E:/test_hel_images/'name='6767_3.png'helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)helmet_features = hog(helmet_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(helmet_features,axis=0))y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))t_start = time.time()svc_new = joblib.load(filename)pre=svc_new.predict(x)print pre
filename='E:/class_SVM_hel_sc.pkl'path='E:/test_hel_images/'name='6767_3.png'helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)helmet_features = hog(helmet_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(helmet_features,axis=0))y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))t_start = time.time()svc_new = joblib.load(filename)pre=svc_new.predict(x)print pre
filename='E:/class_SVM_hel_sc.pkl'path='E:/test_hel_images/'name='6767_3.png'helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)helmet_features = hog(helmet_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(helmet_features,axis=0))y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))#t_start = time.time()#svc_new = joblib.load(filename)#pre=svc_new.predict(x)#print pre
filename='E:/class_SVM_hel_sc.pkl'path='E:/test_hel_images/'name='6767_3.png'helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)helmet_features = hog(helmet,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(helmet_features,axis=0))y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))#t_start = time.time()#svc_new = joblib.load(filename)#pre=svc_new.predict(x)#print pre

##---(Wed May 09 10:34:09 2018)---
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')
unscaled_x = np.vstack((helmet_features, non_helmet_features)).astype(np.float64)scaler = StandardScaler().fit(unscaled_x)
filename='E:/class_SVM_hel_sc.pkl'path='E:/test_hel_images/'name='6767_3.png'helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)helmet_features = hog(helmet,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(helmet_features,axis=0))y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))t_start = time.time()svc_new = joblib.load(filename)pre=svc_new.predict(x)print pre
path_new='E:/test_images/'img_new=cv2.imread(os.path.join(path_new,name),cv2.IMREAD_GRAYSCALE)image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    new_img=cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()else:    new_img=cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')
runfile('C:/Users/DELL API/.spyder/test_bike.py', wdir='C:/Users/DELL API/.spyder')
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')
filename='E:/class_SVM_hel_sc.pkl'path='E:/test_hel_images/'name='5282_2.png'helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)helmet_features = hog(helmet,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(helmet_features,axis=0))y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))t_start = time.time()svc_new = joblib.load(filename)pre=svc_new.predict(x)print pre#x = scaler.transform(unscaled_x)#y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))###t_start = time.time()#svc_new = joblib.load(filename)#pre=svc_new.predict(x)#print pre#path='E:/test_images/'#name='count_1.png'#vehicle= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#vehicle_image = cv2.resize(vehicle, (120,200), interpolation = cv2.INTER_CUBIC)#vehicle_features = hog(vehicle_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)##x = scaler.transform(np.expand_dims(vehicle_features,axis=0))#y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles)))##t_start = time.time()#svc_new = joblib.load(filename)#pre=svc_new.predict(x)#print prepath_new='E:/test_images/'img_new=cv2.imread(os.path.join(path_new,name),cv2.IMREAD_GRAYSCALE)image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    new_img=cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()else:    new_img=cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()
filename='E:/class_SVM_hel_sc.pkl'path='E:/test_hel_images/'name='5828_2.png'helmet= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#helmet_image = cv2.resize(helmet, (120,200), interpolation = cv2.INTER_CUBIC)helmet_features = hog(helmet,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)x = scaler.transform(np.expand_dims(helmet_features,axis=0))y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))t_start = time.time()svc_new = joblib.load(filename)pre=svc_new.predict(x)print pre#x = scaler.transform(unscaled_x)#y = np.hstack((np.ones(total_helmet), np.zeros(total_nonhelmet)))###t_start = time.time()#svc_new = joblib.load(filename)#pre=svc_new.predict(x)#print pre#path='E:/test_images/'#name='count_1.png'#vehicle= cv2.imread(os.path.join(path,name),cv2.IMREAD_GRAYSCALE)#vehicle_image = cv2.resize(vehicle, (120,200), interpolation = cv2.INTER_CUBIC)#vehicle_features = hog(vehicle_image,orientations = 9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)##x = scaler.transform(np.expand_dims(vehicle_features,axis=0))#y = np.hstack((np.ones(total_vehicles), np.zeros(total_nonvehicles)))##t_start = time.time()#svc_new = joblib.load(filename)#pre=svc_new.predict(x)#print prepath_new='E:/test_images/'img_new=cv2.imread(os.path.join(path_new,name),cv2.IMREAD_GRAYSCALE)image1=np.stack((img_new,)*3,-1)font = cv2.FONT_HERSHEY_SIMPLEXif pre[0]==1.0:    new_img=cv2.putText(image1,'YES',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()else:    new_img=cv2.putText(image1,'NO',(50,50), font, 0.5,(0,0,255),2,cv2.LINE_AA)    cv2.imshow('image',new_img)    cv2.waitKey(0)    cv2.destroyAllWindows()
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')

##---(Wed May 09 16:58:40 2018)---
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')

##---(Wed May 09 17:09:15 2018)---
runfile('C:/Users/DELL API/.spyder/test.py', wdir='C:/Users/DELL API/.spyder')