import pandas as pd
from torch.utils import data
import numpy as np
from PIL import Image
import os
import torch



def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)


class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)
    
    def img_normalize(self, img):
        
        a = np.zeros((img.shape))

        a[0,:,:] = img[0,:,:]-np.min(img[0,:,:]) / np.max(img[0,:,:])-np.min(img[0,:,:])
        a[1,:,:] = img[1,:,:]-np.min(img[1,:,:]) / np.max(img[1,:,:])-np.min(img[1,:,:])
        a[2,:,:] = img[2,:,:]-np.min(img[2,:,:]) / np.max(img[2,:,:])-np.min(img[2,:,:])
        
        return a

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """

        # step1
        path = self.root + 'data/'+self.img_name[index]+'.jpeg'
        assert os.path.exists(path), "file not found: {}".format(path)
        img_file = Image.open(path).resize((224,224),Image.ANTIALIAS)

        img = np.transpose(img_file,[2,0,1])
        img = self.img_normalize(img)
        
        # step2
        label = self.label[index]

        return img,label
    
# test = RetinopathyLoader(root,'test')
# image,label = test[0]
class flower(data.Dataset):
    def __init__(self):
        self.train=pd.read_csv('train.csv')
        self.num_classes=len(set(self.train['image_id']))
    def img_normalize(self, img):     
        a = np.zeros((img.shape))
        a[0,:,:] = img[0,:,:]-np.min(img[0,:,:]) / np.max(img[0,:,:])-np.min(img[0,:,:])
        a[1,:,:] = img[1,:,:]-np.min(img[1,:,:]) / np.max(img[1,:,:])-np.min(img[1,:,:])
        a[2,:,:] = img[2,:,:]-np.min(img[2,:,:]) / np.max(img[2,:,:])-np.min(img[2,:,:])      
        return a
    def __getitem__(self,index):
        path=self.train.iloc[index]['file_name']
        assert os.path.exists(path), "file not found: {}".format(path)
        img_file = Image.open(path).resize((224,224),Image.ANTIALIAS)
        img = np.transpose(img_file,[2,0,1])
        img = self.img_normalize(img)
        label=self.train.iloc[index]['image_id']
        return img,label
    def __len__(self):
    ##############################################
    ### Indicate the total size of the dataset
    ##############################################
        return len(self.train)
