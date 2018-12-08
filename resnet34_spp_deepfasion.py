 -*- coding: utf-8 -*-
"""
Created on Sat Aug  18 21:51:58 2018

@author: SaulZhang
"""
from __future__ import print_function, division
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import nn
from torch.nn import functional as F
import torch as t
import math
import random

add_path = '/home/hk/Desktop/Consumer-to-shop_Clothes_Retrieval_Benchmark/imgzip'#the prefix of image's path

class ResidualBlock(nn.Module):
    '''step_size
    实现子Module:ResidualBlock
    '''
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock,self).__init__()
        self.left=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 3, stride, 1 , bias=False),
                nn.BatchNorm2d(outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, 3,1,1,bias=False),
                nn.BatchNorm2d(outchannel))
        self.right=shortcut

    def forward(self,x):
        out=self.left(x)
        residual=x if self.right is None else self.right(x)
        out+=residual
        return F.relu(out)

class ResNet(nn.Module):
    '''
    实现主Module:ResNet34
    ResNet34包含多个layer，每个layer又包含多个residal block
    用子module实现residual block，用_make_layer函数实现layer
    '''
    def __init__(self, num_classes=20,batch_size=32):
        super(ResNet,self).__init__()
        #前几层图像转换
        self.spp_output_num = [7,4,3,2,1]
        self.pre=nn.Sequential(
                nn.Conv2d(3,64,7,2,3,bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3,2,1)
                )
        #重复的layer，分别有3,4,6,3个residual block
        self.layer1=self._make_layer(64, 128 ,  3)
        self.layer2=self._make_layer(128, 256 , 4, stride=2)
        self.layer3=self._make_layer(256, 512 , 6, stride=2)
        self.layer4=self._make_layer(512, 512 , 3, stride=2)
        # self.dropout1 = nn.Dropout(0.5)
        #分类用的全连接
        self.fc1=nn.Linear(512*(1*1+2*2+3*3+4*4+7*7),256)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc2=nn.Linear(256,num_classes)
        self.relu = nn.ReLU(True)
        self.batch_size = batch_size

        # self.dropout2 = nn.Dropout(0.8)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        '''
        构建layer，包含多个residual
        '''
        shortcut=nn.Sequential(
                nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
                nn.BatchNorm2d(outchannel))

        layers=[]
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def spatial_pyramid_pool(self,previous_conv, num_sample, previous_conv_size, out_pool_size):
        '''
        previous_conv: a tensor vector of previous convolution layer
        num_sample: an int number of image in the batch
        previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
        out_pool_size: a int vector of expected output size of max pooling layer
        
        returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
        '''    
        # print(previous_conv.size())
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
            w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            h_pad = int((math.ceil(h_wid*out_pool_size[i] - previous_conv_size[0] + 1)/2))
            w_pad = int(math.ceil(w_wid*out_pool_size[i] - previous_conv_size[1] + 1)/2)
            maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
            # print("type(previous_conv):",*list(previous_conv.children()))
            x = maxpool(previous_conv)
            if(i == 0):
                spp = x.view(num_sample,-1)
                # print("spp size:",spp.size())
            else:
                # print("size:",spp.size())
                spp = torch.cat((spp,x.view(num_sample,-1)), 1)
        return spp

    def forward(self,x):
        # print(x.size(0))
        x=self.pre(x)
        x=self.layer1(x) 
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        # print("type(x):",type(x))
        x=self.spatial_pyramid_pool(previous_conv=x,num_sample=int(x.size(0)),previous_conv_size=[int(x.size(2)),int(x.size(3))],out_pool_size=self.spp_output_num)
        # x=F.avg_pool2d(x,7)
        # x=x.view(x.size(0),-1)
        # x=self.relu(x)
        # x=self.dropout1(x)
        x=self.fc1(x)
        x=self.relu(x)
        # x=self.dropout2(x)
        x = self.fc2(x)
        # x=self.relu(x)
        return x



def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)
def vis_landmark_loss(visibility, landmark, visibility_labels, landmark_labels,batch_size):

    criterion = nn.MSELoss()
    loss1 = criterion(landmark,landmark_labels)
    visibility_labels1 = torch.max(visibility_labels[:,0:2+1],1)[1]
    visibility_labels2 = torch.max(visibility_labels[:,3:5+1],1)[1]
    visibility_labels3 = torch.max(visibility_labels[:,6:8+1],1)[1]
    visibility_labels4 = torch.max(visibility_labels[:,9:11+1],1)[1]

    criterion1 = nn.CrossEntropyLoss()

    loss2_1 = criterion1(visibility[:,0:2+1],visibility_labels1)#torch.max(torch.from_numpy(visibility_labels[:,9:11+1]), 1)
    
    criterion2= nn.CrossEntropyLoss()
    loss2_2 = criterion2(visibility[:,3:5+1],visibility_labels2)

    criterion3 = nn.CrossEntropyLoss()
    loss2_3 = criterion3(visibility[:,6:8+1],visibility_labels3)

    criterion4 = nn.CrossEntropyLoss()
    loss2_4 = criterion4(visibility[:,9:11+1],visibility_labels4)

    loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4

    return loss1+loss2,loss2

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset = '', data_transforms=None, loader = default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            # print("lines,",lines)
            self.img_name = [os.path.join(img_path, line.strip().split(',')[0][1:]) for line in lines]
            self.img_label = []#(0 1 2  3  4  5  6  7  8  9  10 11 12 13)
                               # a b v1 x1 y1 v2 x2 y2 v3 x3 y3 v4 x4 y4  
            for line in lines: #labels(1,20)   v1(0,1,2) v2(3,4,5) v3(6,7,8) v4(9,10,11) l1(12,13) l2(14,15) l3(16,17) l4(18,19) 
                line = line.strip().split(',')[1:]
                # print("line:",line)
                list_each_sample = []
                if int(line[2]) == 0:
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                elif int(line[2]) == 1:
                    list_each_sample.append(0)
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                elif int(line[2]) == 2:
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                    list_each_sample.append(1)

                if int(line[5]) == 0:
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                elif int(line[5]) == 1:
                    list_each_sample.append(0)
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                elif int(line[5]) == 2:
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                    list_each_sample.append(1)

                if int(line[8]) == 0:
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                elif int(line[8]) == 1:
                    list_each_sample.append(0)
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                elif int(line[8]) == 2:
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                    list_each_sample.append(1)

                if int(line[11]) == 0:
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                elif int(line[11]) == 1:
                    list_each_sample.append(0)
                    list_each_sample.append(1)
                    list_each_sample.append(0)
                elif int(line[11]) == 2:
                    list_each_sample.append(0)
                    list_each_sample.append(0)
                    list_each_sample.append(1) 

                list_each_sample.append(int(line[3]))
                list_each_sample.append(int(line[4]))

                list_each_sample.append(int(line[6]))
                list_each_sample.append(int(line[7]))

                list_each_sample.append(int(line[9]))
                list_each_sample.append(int(line[10]))
        
                list_each_sample.append(int(line[12]))
                list_each_sample.append(int(line[13]))
                self.img_label.append(list_each_sample)
 
            print("img_name ",txt_path," ",self.img_name.__len__()) 
            print("img_label ",txt_path," ",self.img_label.__len__()) 
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        img = self.loader(add_path+img_name)

        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))

        return img, label,img_name

# class Net(nn.Module):
#     def __init__(self, num_output=20):
#         super(Net, self).__init__()
#         self.net = nn.Sequential(*list(models.vgg16(pretrained=True).children())[:-3]).cuda()
#         self.pool_layer = nn.MaxPool2d(2).cuda()
#         self.lower = nn.Sequential(
#             nn.Linear(int(512*7*7*3/2),512),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(512,128),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(128,num_output),
#         )

#     def forward(self, x):

#         x = self.net(x)
#         x = self.pool_layer(x)
#         x = x.view(x.size(0),-1)
#         x = self.lower(x)
#         return x


def train_model(model, criterion, optimizer, scheduler ,batch_size,num_epochs):
    since = time.time()

    # best_model_wts = model.state_dict()
    # best_acc = 0.0

    model.load_state_dict(torch.load("./model/ResNet34_DeepFashion_with_dropout_augumentation19.pth"))
    %#SPP:ResNet34_DeepFashion_with_dropout_augumentation79.pth  ============>  val4_bbox_resize_pad Total Loss: 5.7921 visibility classify accurancy: 0.7427  loss2:2.9656 DPL acc: 
    %#SPP:ResNet34_DeepFashion_with_dropout_augumentation59.pth  ============>  val4_bbox_resize_pad Total Loss: 5.7900 visibility classify accurancy: 0.7427  loss2:2.9644 DPL acc: 
    %#SPP:ResNet34_DeepFashion_with_dropout_augumentation39.pth  ============>  val4_bbox_resize_pad Total Loss: 5.7953 visibility classify accurancy: 0.7427  loss2:2.9647 DPL acc: 
    %#SPP:ResNet34_DeepFashion_with_dropout_augumentation19.pth  ============>  val4_bbox_resize_pad Total Loss: 5.7914 visibility classify accurancy: 0.7427  loss2:2.9643 DPL acc:0.3905

    %#SPP:ResNet34_DeepFashion_with_dropout_99.pth   ============>  val4_bbox_resize_pad Total Loss: 5.0491 visibility classify accurancy: 0.7530  loss2:3.7472 DPL acc:0.4655
    %#SPP:ResNet34_DeepFashion_with_dropout_79.pth   ============>  val4_bbox_resize_pad Total Loss: 5.1045 visibility classify accurancy: 0.7519  loss2:3.6740 DPL acc:0.4545
    %#SPP:ResNet34_DeepFashion_with_dropout_59.pth   ============>  val4_bbox_resize_pad Total Loss: 5.1107 visibility classify accurancy: 0.7520  loss2:3.3277 DPL acc:0.4262
    %#SPP:ResNet34_DeepFashion_with_dropout_39.pth   ============>  val4_bbox_resize_pad Total Loss: 5.2963 visibility classify accurancy: 0.7421  loss2:3.0654 DPL acc:0.3568
    %#SPP:ResNet34_DeepFashion_with_dropout_19.pth   ============>  val4_bbox_resize_pad Total Loss: 8.0207 visibility classify accurancy: 0.6797  loss2:3.9003 DPL acc:0.2218

    #drop(0.5,0.8),(512->512->20)ResNet34_DeepFashion_with_dropout_59.pth   ============>  val4_bbox_resize_pad Total Loss: 5.1877 visibility classify accurancy: 0.7641  loss2:3.8764 DPL acc:0.6441
    #drop(0.5,0.8),(512->512->20)ResNet34_DeepFashion_with_dropout_39.pth   ============>  val4_bbox_resize_pad Total Loss: 5.2627 visibility classify accurancy: 0.7491  loss2:4.0264 DPL acc:0.6038
    %#drop(0.5,0.8),(512->512->20)ResNet34_DeepFashion_with_dropout_19.pth   ============>  val4_bbox_resize_pad Total Loss: 5.4924 visibility classify accurancy: 0.7502  loss2:03.5689 DPL acc:0.5127

    #(512->20)ResNet34_DeepFashion_19.pth   ============>  val4_bbox_resize_pad Total Loss: 5.4797 visibility classify accurancy: 0.7646  loss2:3.1224 DPL acc:0.6296
    #(512->20)ResNet34_DeepFashion_39.pth   ============>  val4_bbox_resize_pad Total Loss: 5.0818 visibility classify accurancy: 0.7941  loss2:5.1491 DPL acc:0.6826
    #(512->20)ResNet34_DeepFashion_59.pth   ============>  val4_bbox_resize_pad Total Loss: 5.0308 visibility classify accurancy: 0.7936  loss2:5.8623 DPL acc:0.6920
    #(512->20)ResNet34_DeepFashion_79.pth   ============>  val4_bbox_resize_pad Total Loss: 5.2158 visibility classify accurancy: 0.7830  loss2:4.4876 DPL acc:0.6993
                                                        # test4_bbox_resize_pad Total Loss: 5.2333 visibility classify accurancy: 0.7876  loss2:4.5205 DPL acc:0.7076
    
    #(512->20)ResNet34_DeepFashion_99.pth   ============>  val4_bbox_resize_pad Total Loss: 5.0014 visibility classify accurancy: 0.7894  loss2:5.7575 DPL acc:0.7137
                           #(visibility=2=>landmark->0)  # val4_bbox_resize_pad Total Loss: 5.1960 visibility classify accurancy: 0.7894  loss2:5.7594 DPL acc:0.7266
                                                         #test4_bbox_resize_pad Total Loss: 5.0349 visibility classify accurancy: 0.7953  loss2:5.7725 DPL acc:0.7211
                           #(visibility=2=>landmark->0)  #test4_bbox_resize_pad Total Loss: 5.1741 visibility classify accurancy: 0.7953  loss2:5.7729 DPL acc:0.7346
                           
    #(512->20)ResNet34_DeepFashion_99.pth   ============>  val4_bbox_resize_pad Total Loss: 5.3005 visibility classify accurancy: 0.8069  loss2:4.1453 DPL acc:0.7354 score:0.0977
                                                         #test4_bbox_resize_pad Total Loss: 5.3260 visibility classify accurancy: 0.8116  loss2:4.2318 DPL acc:0.7458 score:0.0961



    #VGG16:           DeepFashion_686.pth   ============>  val4_bbox_resize_pad Total Loss: 9.8817 visibility classify accurancy: 0.6593  loss2:3.2009 DPL acc:0.1581
    #VGG16:           DeepFashion_784.pth   ============>  val4_bbox_resize_pad Total Loss: 9.8817 visibility classify accurancy: 0.6593  loss2:3.2009 DPL acc:0.1581


    epoch_record = []
    train_loss_record = []
    val_loss_record = []
    for epoch in range(0,1):#num_epochs
        epoch_record.append(epoch)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('*' * 20)

        # Each epoch has a training and validation phase
        for phase in ['train4_bbox_resize_pad', 'val4_bbox_resize_pad']:

            if phase == 'train4_bbox_resize_pad':
                continue
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.eval()
                # model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_loss2 = 0.0
            # running_corrects = 0
            k = 0
            # Iterate over data.
            since_epoch = time.time()

            num_correct_batch_visibility = 0.
            total_num_correct = 0
            total_correct_num_landmark = 0

            for data in dataloders[phase]:
                
                # print("=",end='')
                k += 1
                # get the inputs
                inputs, labels , img_name = data

                nlabels = []
                for lab in labels:
                    nlabels.append(lab.tolist())
                labels = nlabels

                labels = list(map(list,(zip(*labels))))#矩阵转置
                labels = torch.from_numpy(np.array(labels))

                x = torch.FloatTensor([1])
                labels = labels.type_as(x)#将labels的类型转化为torch.FloatTensor
                labels = labels.cuda()

                if torch.cuda.is_available():
                    inputs = Variable(inputs).cuda()
                    labels = Variable(labels).cuda()
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                outputs = model(inputs)

                visibility = outputs[:,0:12]

                landmark =  outputs[:,12:]

                visibility_labels = labels[:,0:12]

                landmark_labels =   labels[:,12:]

                if phase == "val4_bbox_resize_pad":
                    # print(landmark.size())
                    # print(landmark[:,0:1+1].size())

                    # print("*&*&"*10,torch.sum((landmark[:,0:1+1] - landmark_labels[:,0:1+1])**2,1))
                    # break
                    pos1 = torch.sqrt(torch.sum((landmark[:,0:1+1] - landmark_labels[:,0:1+1])**2,1))
                    pos2 = torch.sqrt(torch.sum((landmark[:,2:3+1] - landmark_labels[:,2:3+1])**2,1))
                    pos3 = torch.sqrt(torch.sum((landmark[:,4:5+1] - landmark_labels[:,4:5+1])**2,1))
                    pos4 = torch.sqrt(torch.sum((landmark[:,6:7+1] - landmark_labels[:,6:7+1])**2,1))
                    # print("pos1:",pos1)
                    # print("pos2:",pos2)
                    # print("pos3:",pos3)
                    # print("pos4:",pos4)

                    correct_num_pos1 = (pos1 < (35*224/512)).sum() 
                    correct_num_pos2 = (pos2 < (35*224/512)).sum() 
                    correct_num_pos3 = (pos3 < (35*224/512)).sum() 
                    correct_num_pos4 = (pos4 < (35*224/512)).sum() 
                    # print("correct_num_pos1:",correct_num_pos1.item())
                    # print("correct_num_pos2:",correct_num_pos2.item())
                    # print("correct_num_pos3:",correct_num_pos3.item())
                    # print("correct_num_pos4:",correct_num_pos4.item())
                    total_correct_num_landmark += correct_num_pos1.item() + correct_num_pos2.item() + correct_num_pos3.item() +correct_num_pos4.item()
                    # print("total_correct_num_landmark:",total_correct_num_landmark.item())

                _, pred_visibility1 = torch.max(visibility[:,0:2+1], 1)
                
                
                _, pred_visibility2 = torch.max(visibility[:,3:5+1], 1)
                
                _, pred_visibility3 = torch.max(visibility[:,6:8+1], 1)
                
                _, pred_visibility4 = torch.max(visibility[:,9:11+1], 1)


                _,label_visibility1 = torch.max(visibility_labels[:,0:2+1], 1)

                _,label_visibility2 = torch.max(visibility_labels[:,3:5+1], 1)

                _,label_visibility3 = torch.max(visibility_labels[:,6:8+1], 1)

                _,label_visibility4 = torch.max(visibility_labels[:,9:11+1], 1)

                # print("pred_visibility",pred_visibility)
                # print("label_visibility",label_visibility)

                num_correct_batch_visibility = (pred_visibility1 == label_visibility1).sum() +(pred_visibility2 == label_visibility2).sum()+(pred_visibility3 == label_visibility3).sum()+(pred_visibility4 == label_visibility4).sum()

                # print(num_correct)

                total_num_correct += num_correct_batch_visibility.item()

                landmark1 = landmark.cpu().detach().numpy()
                landmark_labels1 = landmark_labels.cpu().detach().numpy()

                if k % 200 == 0 or (phase == "val4_bbox_resize_pad" and k % 20 == 0) :
                    count = 0
                    for (img,land,land_label) in zip(img_name,landmark1,landmark_labels1):
                        count += 1
                        if count % 8 != 0:
                            continue
                        img = Image.open(add_path+img)
                        plt.imshow(img,cmap=plt.gray())
                        plt.scatter(land[0],land[1],c='g')
                        plt.scatter(land[2],land[3],c='g')
                        plt.scatter(land[4],land[5],c='g')
                        plt.scatter(land[6],land[7],c='g')

                        plt.scatter(land_label[0],land_label[1],c='r')
                        plt.scatter(land_label[2],land_label[3],c='r')
                        plt.scatter(land_label[4],land_label[5],c='r')
                        plt.scatter(land_label[6],land_label[7],c='r')   
                                             
                        plt.savefig("./record_picture/save_img_pad_resnet_with_dropout_SPP_augumentation/"+phase[0:5]+'-'+str(epoch)+'-'+str(k)+'-'+str(count)+'.jpg')
                        plt.close()
                
                
                loss,loss2 = vis_landmark_loss(visibility, landmark, visibility_labels, landmark_labels,batch_size)
                optimizer.zero_grad()

                running_loss += float(loss)

                running_loss2 += float(loss2)
                # print("float(loss):",float(loss))
                
                # if k%100 == 0:
                #     print(k," loss:",running_loss/(k))

                # backward + optimize only if in training phase
                if phase == 'train4_bbox_resize_pad':
                    loss.backward()
                    optimizer.step()
                # print("total_correct_num_landmark:",total_correct_num_landmark)
            if (epoch + 1) % 20 == 0: 
                torch.save(model.state_dict(), './model/ResNet34_DeepFashion_with_dropout_augumentation'+str(epoch)+'.pth')
                print("epoch_record",epoch_record)
                print("train_loss_record",train_loss_record)
                print("val_loss_record",val_loss_record)

                print()
                try:
                    print("save the model "+"./model/ResNet34_DeepFashion_with_dropout_augumentation"+str(epoch)+".pth")
                except:
                    pass

            epoch_loss = running_loss / (dataset_sizes[phase]/batch_size)
            epoch_loss2 = running_loss2 / (dataset_sizes[phase]/batch_size)

            if phase == 'train4_bbox_resize_pad':
                train_loss_record.append(epoch_loss/224)
            else:
                val_loss_record.append(epoch_loss/224)

            print('{} Total Loss: {:.4f} visibility classify accurancy: {:.4f}  loss2:{:.4f} DPL acc:{:.4f}'.format(phase, epoch_loss/224,total_num_correct/(dataset_sizes[phase]*4),
                                                                                                                                        epoch_loss2,total_correct_num_landmark/(dataset_sizes[phase]*4)))

            print('Spend {}s in this epoch'.format(time.time() - since_epoch))
            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = model.state_dict()
        # break
    '''
    plt.figure()
    plt.plot(epoch_record,train_loss_record,'m-x',linewidth=1,label="train_loss")
    plt.plot(epoch_record,val_loss_record,'g-o',linewidth=1,label="test_loss")
    plt.ylim(0.00,30.00)
    plt.xlim(0,100)
    plt.xlabel("epoch")
    plt.ylabel("train/val loss")
    plt.title("ResNet34_Loss-chart")
    # plt.show()
    plt.legend(loc='upper left',title='legend') 
    plt.savefig("ResNet34_Loss-chart.jpg")
    '''
    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return model

if __name__ == '__main__':
    data_transforms = {
    'train4_bbox_resize_pad': transforms.Compose([
        # transforms.Resize((256,256)),
        # transforms.RandomSizedCrop(224),
        # transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(contrast=random.uniform(0,1),brightness=random.uniform(0,1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'val4_bbox_resize_pad': transforms.Compose([
        # transforms.Resize((256,256)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test4_bbox_resize_pad': transforms.Compose([
    # transforms.Resize((256,256)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    }
    image_datasets = {x: customData(img_path='/home/hk/Desktop/Consumer-to-shop_Clothes_Retrieval_Benchmark/imgzip',
                                   txt_path=('./divdata/' + x + '.csv'),
                                   data_transforms=data_transforms,
                                   dataset=x) for x in ['train4_bbox_resize_pad', 'val4_bbox_resize_pad','test4_bbox_resize_pad']}
    # print(image_datasets)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train4_bbox_resize_pad', 'val4_bbox_resize_pad','test4_bbox_resize_pad']}
    # print(dataset_sizes['train4_bbox_resize_pad'])

    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                             batch_size=32,
                                             shuffle=True,
                                             num_workers=4) for x in ['train4_bbox_resize_pad', 'val4_bbox_resize_pad','test4_bbox_resize_pad']}

    resnet = ResNet(num_classes=20,batch_size=32)
    if torch.cuda.is_available():
        resnet = resnet.cuda()

    criterion = nn.MSELoss()
    optimizer_ft = torch.optim.Adam(resnet.parameters(), lr=1e-2)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=3, gamma=0.9)
    model_ft = train_model(model=resnet,
                       criterion=criterion,
                       optimizer=optimizer_ft,
                       scheduler=exp_lr_scheduler,
                       batch_size=32,
                       num_epochs=100)

#Reference ResNet:https://blog.csdn.net/zhenaoxi1077/article/details/80951034
#Related Paper:https://arxiv.org/abs/1512.03385
