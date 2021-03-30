import os,shutil
from keras import layers
from keras import models
from keras import utils
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input,decode_predictions
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 

#original_dataset_dir = r'D:\python源程序\猫狗图片\train'
base_dir = r'D:\python源程序\小猫狗图片'

train_dir = base_dir + r'\train'
test_dir = base_dir + r'\test'
validation_dir = base_dir + r'\validation'

train_cats_dir = train_dir + r'\cats'
train_dogs_dir = train_dir + r'\dogs'

validation_cats_dir = validation_dir + r'\cats'
validation_dogs_dir = validation_dir + r'\dogs'

test_cats_dir = test_dir + r'\cats'
test_dogs_dir = test_dir + r'\dogs'

#fnames = list('dog.{}.jpg'.format(i) for i in range(1500,2000))
'''
model = models.Sequential()
model.add(layers.Conv2D(32,[3,3],activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,[3,3],activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,[3,3],activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,[3,3],activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])
'''
model = models.load_model(r'D:\python源程序\神经网络\训练模型\cats_and_dogs_small_2.h5')
'''
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)
'''

'''
history_dict = history.history
acc = history_dict['acc']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation accuracy')
plt.legend()

plt.show()
'''
#model.save(r'D:\python源程序\cats_and_dogs_small_2.h5')

'''
test_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)
'''
img_path = test_dogs_dir + r'\dog.1511.jpg'
#im = Image.open(img_path)
img = image.load_img(img_path,target_size=(150,150))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

#preds = model.predict_classes(x,verbose=0)
#print(preds)
'''
i = 0
for x,y in validation_generator:
    if i == 10:
        break
    preds = model.predict(x)
    preds = np.argmax(preds)
    print(preds)
    i += 1
'''
'''
if preds > 0.8:
    print('狗')
elif preds < 0.3:
    print('猫')
else:
    print('不能识别')
'''
#im.show()
