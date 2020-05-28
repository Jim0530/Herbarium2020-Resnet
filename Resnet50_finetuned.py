import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization
import os
import time
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]="2"
train=pd.read_csv('train.csv')
count=len(train)
batch_size=128
test=pd.read_csv('test.csv')
print('reading finished')
t1=time.time()
print('start=',time.ctime(t1))
data_generator = ImageDataGenerator(featurewise_std_normalization=True)
train_generator=data_generator.flow_from_dataframe(dataframe=train,directory='train',
                                                      x_col="file_name",
                                                      y_col="category_name",
                                                      target_size=(224,224),
                                                      batch_size=128,
                                                      class_mode="categorical")
test_generator=data_generator.flow_from_dataframe(dataframe=test,batch_size=128,directory="test",x_col="file_name",target_size=(224,224),class_mode=None,shuffle=False)
t2=time.time()
print('time=',t2-t1)
print('data_generator constructed')
num_classes = len(train_generator.class_indices)
model = Sequential()
model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))
model.add(BatchNormalization())
model.add(Dense(num_classes, activation='softmax'))
model.layers[0].trainable = False
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit_generator(
        train_generator,
        steps_per_epoch=int(count/batch_size) + 1,
        epochs=3,
        verbose=1)
print('training finished\nstrat predicting')
model.save('my_model.h5')
answer=model.predict_generator(test_generator,verbose=1)
answer=np.argmax(answer,axis=1)
np.save('answer.npy')