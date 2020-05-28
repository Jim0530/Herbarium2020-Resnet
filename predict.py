from keras.models import load_model
import json
import os
import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.image import ImageDataGenerator
os.environ["CUDA_VISIBLE_DEVICES"]="2"
data_generator = ImageDataGenerator(featurewise_std_normalization=True)
test=pd.read_csv('test.csv')
test_generator=data_generator.flow_from_dataframe(dataframe=test,batch_size=128,directory="test",x_col="file_name",target_size=(224,224),class_mode=None,shuffle=False)
model=load_model('my_model.h5')
answer=model.predict_generator(test_generator,verbose=1)
answer=np.argmax(answer,axis=1)
with open('answer.pkl','wb') as f:
	pickle.dump(answer,f)
