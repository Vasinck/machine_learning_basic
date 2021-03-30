from keras import layers
from keras import models
from keras import utils
from keras.preprocessing import image
import numpy as np 
import os 

def load_data():
    path = r'D:\python源程序\小猫狗图片\train\cats' + '//'
    files = os.listdir(path)
    images = []
    labels = []
    for f in files:
        img_path = path + f
        img = image.load_img(img_path, target_size=(150,150))
        img_array = image.img_to_array(img)
        images.append(img_array)
        labels.append(1)
    path = r'D:\python源程序\小猫狗图片\train\dogs' + '//'
    files = os.listdir(path)
    for f in files:
        img_path = path + f
        img = image.load_img(img_path, target_size=(150,150))
        img_array = image.img_to_array(img)
        images.append(img_array)
        labels.append(0)

    data = np.array(images)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, 2)
    return data, labels

datas,labels = load_data()