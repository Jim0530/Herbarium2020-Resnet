#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:33:23 2020

@author: user
"""
from dataloader import flower
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import tensorboard_logger as tb_log
from sklearn.metrics import confusion_matrix
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
        
class BasicBlock(nn.Module):
    def __init__(self,insize,outsize,stride = 1):
        super(BasicBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(insize, outsize, kernel_size = 3, stride = stride, padding = 1, bias = False),
            nn.BatchNorm2d(outsize),
            nn.ReLU(inplace=True),
            nn.Conv2d(outsize, outsize, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(outsize)
            )
        self.downsample = nn.Sequential()
        if stride !=1 or insize != outsize:
            self.downsample = nn.Sequential(
                nn.Conv2d(insize, outsize, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(outsize)
                )
        
    def forward(self,x):
        residual = x
        out = self.block(x)
        
        residual = self.downsample(x)
            
        out+=residual
        out = F.relu(out)
        
        return out
        
class BottleneckBlock(nn.Module):
    def __init__(self,insize,outsize,stride=1):
        super(BottleneckBlock,self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(insize,outsize,kernel_size=1,stride = 1, bias = False),
            nn.BatchNorm2d(outsize),
            nn.ReLU(inplace=True),
            nn.Conv2d(outsize, outsize, kernel_size=3,stride = stride,padding = 1,bias = False),
            nn.BatchNorm2d(outsize),
            nn.ReLU(inplace=True),
            nn.Conv2d(outsize,outsize*4,kernel_size=1,stride = 1, bias = False),
            nn.BatchNorm2d(outsize*4)
            )
        self.downsample = nn.Sequential()
        if stride !=1 or insize ==64 :
            self.downsample = nn.Sequential(
                nn.Conv2d(insize, outsize*4, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(outsize*4)
                )
    def forward(self,x):
        residual = x 
        out = self.block(x)
        
        residual = self.downsample(x)
            
        out+=residual
        out = F.relu(out)
        
        return out
    
class ResNet18(nn.Module):
    def __init__(self,num_class,size = [3,64,64,128,256,512,5],stride = [2,1,2,2,2]):
        super(ResNet18,self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(size[0],size[1],kernel_size=7,stride=stride[0],padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False)
            )
        self.basiclayer = nn.ModuleList()
        
        for i in range(1,5):
            self.basiclayer.append(BasicBlock(size[i],size[i+1],stride=stride[i]))
            self.basiclayer.append(BasicBlock(size[i+1], size[i+1], 1))
        
        self.layer1 = nn.Sequential(
            self.basiclayer[0],
            self.basiclayer[1]
            )
        
        self.layer2 = nn.Sequential(
            self.basiclayer[2],
            self.basiclayer[3]
            )
        
        self.layer3 = nn.Sequential(
            self.basiclayer[4],
            self.basiclayer[5]
            )
            
        self.layer4 = nn.Sequential(
            self.basiclayer[6],
            self.basiclayer[7]
            )
        
        self.classify=nn.Sequential(
            nn.Linear(in_features=512, out_features=num_class, bias=True)
            )
          
    def forward(self,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,kernel_size=7,stride=1,padding=0)
        x = x.view(-1,self.classify[0].in_features)
        x = self.classify(x)
        
        return x
       
class ResNet50(nn.Module):
     def __init__(self,num_class = 5,size = [3,64,64,128,256,512,5],stride = [2,1,2,2,2]):
         super(ResNet50,self).__init__()
         self.activation = nn.ReLU(inplace=True)
         self.layer0 = nn.Sequential(
            nn.Conv2d(size[0],size[1],kernel_size=7,stride=stride[0],padding=3,bias=False),
            nn.BatchNorm2d(64),
            self.activation,
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False)
            )
        
         self.layer1 = nn.Sequential(
            BottleneckBlock(size[1], size[2],stride = stride[1]),
            BottleneckBlock(size[2]*4, size[2],1),
            BottleneckBlock(size[2]*4, size[2],1)
            )
        
         self.layer2 = nn.Sequential(
            BottleneckBlock(size[2]*4, size[3],stride = stride[2]),
            BottleneckBlock(size[3]*4, size[3],1),
            BottleneckBlock(size[3]*4, size[3],1),
            BottleneckBlock(size[3]*4, size[3],1)
            )
        
         self.layer3 = nn.Sequential(
            BottleneckBlock(size[3]*4, size[4],stride = stride[3]),
            BottleneckBlock(size[4]*4, size[4],1),
            BottleneckBlock(size[4]*4, size[4],1),
            BottleneckBlock(size[4]*4, size[4],1),
            BottleneckBlock(size[4]*4, size[4],1),
            BottleneckBlock(size[4]*4, size[4],1)
            )
        
         self.layer4 = nn.Sequential(
            BottleneckBlock(size[4]*4, size[5],stride = stride[4]),
            BottleneckBlock(size[5]*4, size[5],1),
            BottleneckBlock(size[5]*4, size[5],1)
            )
        
         self.classify = nn.Sequential(
             nn.Linear(in_features=2048, out_features=num_class, bias=True)
            )
     def forward(self ,x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x,kernel_size=7,stride=1,padding=0)
        x = x.view(-1,self.classify[0].in_features)
        x = self.classify(x)
        return x
        
def train(data, label, model, loss_function, optimizer):
    data = data.float().cuda()
    label = label.long()
    
    model.train()
    
    loss = 0
    prediction = model(data).cpu()
    loss = loss_function(prediction, label)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss     

def test(data,label, model):
    data = data.float().cuda()
    label = label.long()
    
    model.eval()
    
    prediction = np.argmax(F.softmax(model(data).cpu(), dim=1).data.numpy(), axis=1)
    acc = np.mean(np.equal(prediction.data,label.data.numpy()))
    
    return acc,  np.asarray(prediction.data), label.data.numpy()

def get_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.Blues):
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)


data_train = flower()  
data_test = flower()   
output_dir = 'output'
tb_log.configure(os.path.join(output_dir, 'tensorboard'))   
cat=data_train.num_classes  
def main():
    network = 1
    batch_size = 32
    total_epoch = 11
    mode_state = "train"
    model_weight = "epoch_10.pkl"
    
    if network == 0:
        model = ResNet18(num_class=cat)
    elif network == 1:
        model = ResNet50(num_class=cat)
    
    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    loss_function = nn.CrossEntropyLoss()
    highest_test_acc = 0
    iterate = 1
    
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=2)
    
    if mode_state == 'train':
        for epoch in range(1,total_epoch):
            for it_r,(batch_data,batch_label) in enumerate(train_loader):
                loss = train(batch_data,batch_label,model,loss_function,optimizer)
                
                tb_log.log_value('cross-entropy_loss', loss, iterate)
                iterate += 1
                
                train_acc = 0
                train_acc = test(batch_data, batch_label, model)
                tb_log.log_value('train_acc', train_acc[0], iterate)
                
                print("Epoch :",'{:04d}'.format(epoch),
                      "\nCrossEntropy Loss :",'{:.6f}'.format(loss.data.numpy()),
                      "train_acc:"," {:.6f}" .format(train_acc[0]))
                
            if epoch % 2 == 0:
                save_name = "./checkpoint_18/epoch_" + str(epoch) + ".pkl"
                torch.save(model.state_dict(), save_name)
                
                test_acc = 0
                count = 1
                for cur_it, (batch_data, batch_label) in enumerate(test_loader):
                    a, _,_ = test(batch_data, batch_label, model)
                    test_acc += a
                    count += 1
                    print("run test:","{:04d}".format(cur_it))
                test_acc = test_acc / count
                tb_log.log_value('test_acc', test_acc, epoch)
                highest_test_acc = max(highest_test_acc, test_acc)
                
                print("test_acc:"," {:.6f} " .format( test_acc),
                      "highest_test_acc:"," {:.6f}" .format (highest_test_acc))
    else:
        
        print("start loading weight...")
        model.load_state_dict(torch.load("./checkpoint_50/" + model_weight))
        print("finish..")
        
        test_loader = DataLoader(data_test, batch_size=batch_size, num_workers=4)
        test_acc = 0

            
        for it, (batch_data, batch_label) in enumerate(test_loader):
            test_a, pred, label = test(batch_data, batch_label, model)
            print("test_acc of No.","{:04d}".format(it)," \n"+str(test_a)+"\n")
            test_acc += test_a
            
            if it == 0:
                total_predict = pred*1
                total_label = label*1
            else:
                total_predict = np.concatenate((total_predict,pred), axis=0)
                total_label = np.concatenate((total_label,label), axis=0)
            
        test_acc = test_acc / (it + 1)

        print("result of resnet50" )
        # print("train_acc: %.4f" % train_acc)
        print("test_acc: %.4f " % test_acc)
        
        y_actu = pd.Series(total_label.tolist(), name='GT')
        y_pred = pd.Series(total_predict.tolist(), name='Prediction')
        df_confusion = pd.crosstab(y_actu, y_pred)
        df_confusion_norm = df_confusion.div(df_confusion.sum(axis=1),axis=0)
        get_confusion_matrix(df_confusion_norm)

        
if __name__ == "__main__":
    main()