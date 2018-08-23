import cv2
import numpy as np
from keras.layers import Dense, Flatten,Dropout
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import Sequential
import os
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD,Adam,RMSprop
from keras import metrics

def image_change(img_rgb):
    if img_rgb.shape[0] >= img_rgb.shape[1]:
        img_vertical = img_rgb
    else:
        img_vertical = np.rot90(img_rgb)
    if img_vertical.shape[0] / 640.0 >= img_vertical.shape[1] / 480.0:
        img_resized_rgb = cv2.resize(img_vertical, (int(640.0 * img_vertical.shape[1] / img_vertical.shape[0]), 640)) # (640, *, 3)
    else:
        img_resized_rgb = cv2.resize(img_vertical, (480, int(480.0 * img_vertical.shape[0] / img_vertical.shape[1]))) # (*, 480, 3)
    return img_resized_rgb

def model():
    model = Sequential()
    ###first converlutional layer
    model.add(Conv2D(96, (15, 10),strides = 5, use_bias = True,input_shape=(640,480,3)))
    model.add(LeakyReLU(alpha=0.3))    
    model.add(MaxPooling2D(pool_size=(4, 3), strides = 2, border_mode = 'valid'))
    model.add(BatchNormalization(axis = 1))
    print('converlutional layer one finised')
    
    model.add(Conv2D(256, (6, 3),strides = 2, use_bias = True))
    model.add(LeakyReLU(alpha=0.3))    
    model.add(MaxPooling2D(pool_size=(3, 3), strides = 2, border_mode = 'valid'))
    model.add(BatchNormalization(axis = 1))
    print('converlutional layer two finised')
    
    model.add(Conv2D(256, (3, 3),strides = 1, padding = 'same',use_bias = True))
    model.add(LeakyReLU(alpha=0.3))
    print('converlutional layer three finised')
    
    model.add(Conv2D(384, (3, 3),strides = 1, padding = 'same',use_bias = True))
    model.add(LeakyReLU(alpha=0.3))
    print('converlutional layer four finised')
    
    model.add(Conv2D(256, (3, 3),strides = 1, padding = 'same',activation='relu',use_bias = True))
    model.add(MaxPooling2D(pool_size=(3, 3), strides = 2, padding = 'valid'))
    print('converlutional layer five finised')
    
    model.add(Flatten())
    
    ###########dense layer
    model.add(Dense(4096, use_bias = True))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(rate = 0.5))
    
    model.add(Dense(4096, use_bias = True))
    model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(rate = 0.5))
    
    model.add(Dense(1000, activation = 'relu',use_bias = True))
    model.add(Dense(3,activation = 'softmax'))
    return model


imgs=[]
y=[]
path_1 = 'train/Type_1'
path_2 = 'train/Type_2'
path_3 = 'train/Type_3'
##img_add = 'D:/UNSW/IMG_1303.JPG'
imgs_1=[]
for str_name_file_or_dir in os.listdir(path_1):
    if ('.jpg' in str_name_file_or_dir) == True:
        imgs_1.append(os.path.join(path_1, str_name_file_or_dir))
        #y.append(1)
        
#imgs_1.sort()
imgs_2=[]
for str_name_file_or_dir in os.listdir(path_2):
    if ('.jpg' in str_name_file_or_dir) == True:
        imgs_2.append(os.path.join(path_2, str_name_file_or_dir))
        
        #y.append(2)
        print('2')
#imgs_2.sort()
imgs_3=[]
for str_name_file_or_dir in os.listdir(path_3):
    if ('.jpg' in str_name_file_or_dir) == True:
        imgs_3.append(os.path.join(path_3, str_name_file_or_dir))
        
        #y.append(3)
        print('3')
#imgs_3.sort()
ii=0
img = []
for i in imgs_1:
    ii=ii+1
    img_rgb = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
    img_changed = image_change(img_rgb)
    if img_changed.shape == (640,480,3):
        img.append(img_changed)
        y.append([1,0,0])
        print(ii)
imgs1 = np.array(img)
    

img = []
for i in imgs_2:
    ii=ii+1
    img_rgb = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
    img_changed = image_change(img_rgb)
    if img_changed.shape == (640,480,3):
        img.append(img_changed)
        y.append([0,1,0])
        print(ii)
imgs2 = np.array(img)

img = []
for i in imgs_3:
    ii=ii+1
    img_rgb = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
    img_changed = image_change(img_rgb)
    if img_changed.shape == (640,480,3):
        img.append(img_changed)
        y.append([0,0,1])
        print(ii)
imgs3 = np.array(img)
imgs = np.vstack((imgs1,imgs2,imgs3))


train_data_y = np.array(y)
train_data_x = np.array(imgs)

model = model()
sgd = SGD(lr=0.01, decay= 0.9, momentum=0.9, nesterov=True)
#sgd = Adam()
model.compile(loss='categorical_crossentropy', optimizer=sgd,
              metrics=['accuracy'])

#model.fit(np.array(img_rgbs),y,batch_size=20, epochs=200)
#a=model.predict(np.array([img_test_x]))
#print(a)
x_train,x_test,y_train,y_test = train_test_split(train_data_x,train_data_y,test_size = 0.2)
h=model.fit(x_train,y_train,batch_size = 20, epochs=200,verbose = 1)
score = model.evaluate(x_test,y_test,batch_size =20,verbose =1)
pre = model.predict(x_test)

print(score)

