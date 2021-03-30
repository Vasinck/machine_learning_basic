from keras import layers
from keras import models
from keras.datasets import imdb
from keras.preprocessing import sequence
import numpy as np 
import matplotlib.pyplot as plt 

max_feature = 10000
max_len = 500

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_feature)
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)

model = models.Sequential()
model.add(layers.Embedding(max_feature,128,input_length=max_len))
model.add(layers.Conv1D(64,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(64,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(64,7,activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.GlobalMaxPooling1D())
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
#model.summary()
model.fit(x_train,y_train,epochs=4,batch_size=128,validation_split=0.2)
print(model.evaluate(x_test,y_test))