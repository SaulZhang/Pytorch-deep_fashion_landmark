from __future__ import print_function, division
from torch.utils.data import Dataset
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F

train_filename = "train4"
val_filename = "val4"
test_filename = "test4"

class myloss(nn.Module):
    # TODO MyLoss
    def __init__(self):
        super(myloss, self).__init__()
        return

    def forward(self, target, output):  # mse：最小平方误差函数
        loss = -(torch.sum(target * (torch.exp(output))) / (torch.sum(torch.exp(output)))).cuda()
        return loss

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

class customData(Dataset):
    def __init__(self, img_path, txt_path, dataset='', data_transforms=None, loader=default_loader):
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            self.img_name = [os.path.join(img_path, line.strip().split(',')[0]) for line in lines]
            self.img_label = []
            self.img_visible = []
            for line in lines:
                cline = line.strip().split(',')[3:]
                cline = list(map(int, cline))
                ncline = []
                vncline = []
                if cline[0] == 0:
                    vncline.append(1)
                    vncline.append(0)
                    vncline.append(0)
                elif cline[0] == 1:
                    vncline.append(0)
                    vncline.append(1)
                    vncline.append(0)
                else:
                    vncline.append(0)
                    vncline.append(0)
                    vncline.append(1)
                ncline.append(cline[1])
                ncline.append(cline[2])
                if cline[3] == 0:
                    vncline.append(1)
                    vncline.append(0)
                    vncline.append(0)
                elif cline[3] == 1:
                    vncline.append(0)
                    vncline.append(1)
                    vncline.append(0)
                else:
                    vncline.append(0)
                    vncline.append(0)
                    vncline.append(1)
                ncline.append(cline[4])
                ncline.append(cline[5])
                if cline[6] == 0:
                    vncline.append(1)
                    vncline.append(0)
                    vncline.append(0)
                elif cline[6] == 1:
                    vncline.append(0)
                    vncline.append(1)
                    vncline.append(0)
                else:
                    vncline.append(0)
                    vncline.append(0)
                    vncline.append(1)
                ncline.append(cline[7])
                ncline.append(cline[8])
                if cline[9] == 0:
                    vncline.append(1)
                    vncline.append(0)
                    vncline.append(0)
                elif cline[9] == 1:
                    vncline.append(0)
                    vncline.append(1)
                    vncline.append(0)
                else:
                    vncline.append(0)
                    vncline.append(0)
                    vncline.append(1)
                ncline.append(cline[10])
                ncline.append(cline[11])
                self.img_label.append(ncline)
                self.img_visible.append(vncline)
            print("img_name ", txt_path, " ", self.img_name.__len__())
            print("img_label ", txt_path, " ", self.img_label.__len__())
        self.data_transforms = data_transforms
        self.dataset = dataset
        self.loader = loader

    def __len__(self):
        return len(self.img_name)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        label = self.img_label[item]
        visible = self.img_visible[item]
        img = self.loader(img_name)
        if self.data_transforms is not None:
            try:
                img = self.data_transforms[self.dataset](img)
            except:
                print("Cannot transform image: {}".format(img_name))
        return img, label, visible, img_name


class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        #
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2]).cuda()
        # for param in self.resnet_layer.parameters():
        #     param.requires_grad = False
        # self.fc = nn.Linear(self.resnet_layer.children()[-1].in_features, 12)
        # self.transion_layer = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=3)
        self.pool_layer = nn.MaxPool2d(2).cuda()
        # for param in self.pool_layer.parameters():
        #     param.requires_grad = False
        self.Linear_layer = nn.Linear(25088, 20).cuda()#vgg16  49152
        self.Cov_layer = nn.Conv2d(2048,512,kernel_size=3).cuda()

        self.localization = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=7),#[4, 512, 7, 7]
            # nn.MaxPool2d(2),
            nn.ReLU(True),

            nn.Conv2d(256, 50, kernel_size=5),#torch.Size([4, 512, 7, 7])
            # nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        ).cuda()

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(1800, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        ).cuda()

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
    def stn(self, x):
        xs = self.localization(x).cuda()#torch.Size([4, 512, 5, 5])
        xs = xs.view(xs.size(0), -1)#xs.view(-1, 512*7*7).cuda()
        theta = self.fc_loc(xs).cuda()
        theta = theta.view(-1, 2, 3).cuda()
        # #
        grid = F.affine_grid(theta, x.size()).cuda()
        x = F.grid_sample(x, grid).cuda()

        return x

    def forward(self, x):
        x = self.resnet_layer(x)
        # x= self.Cov_layer(x)
        # x = self.pool_layer(x)
        # print("resnetshape",x.shape)
        x1 = self.stn(x)
        x2 =self.stn(x)
        x3 = self.stn(x)
        x4 = self.stn(x)
        x = torch.cat((x1,x2,x3,x4),1)
        x = self.Cov_layer(x)
        x = self.pool_layer(x)
        # torch.cat()
        # print(x.shape)
        # x = self.fc(x)
        # x = self.transion_layer(x)
        #
        # print("STN:",x.shape)#8 n256 256
        # x = self.pool_layer(x)
        x = x.view(x.size(0), -1)
        # print("aaa")
        # print(x.shape)

        x = self.Linear_layer(x)

        return x
'''
Epoch 0/199
----------
Mode= val4
100  loss: 68.77475992838542    vloss: 8.184478759765625
200  loss: 43.78034464518229    vloss: 3.8096506595611572
val4 Loss: 4.8489 
******************** 
val4 Total Loss: 4.8489 
visibility classify accurancy: 0.7474   
DPL acc:0.7777 
score:0.0653 
 ********************
Mode= test4
100  loss: 71.37303670247395    vloss: 4.946088790893555
200  loss: 228.1916300455729    vloss: 10.89534854888916
test4 Loss: 5.0943 
******************** 
test4 Total Loss: 5.0943 
visibility classify accurancy: 0.7425   
DPL acc:0.7779 
score:0.0671 
 ********************
'''
#21-16 lock all the landmark point
def train_model(model, criterion, optimizer, scheduler, batch_size, num_epochs):
    since = time.time()
    model_name = "./models_initial/DeepFashion_100.pth"
    if model_name != '':
        print()
        print(model_name,"is loading...")
        print()
        model.load_state_dict(torch.load(model_name))
    
    for epoch in range(1):#num_epochs
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in [train_filename,test_filename]:  # ,'test4_transform_bbox']:
            
            if phase == train_filename:
                continue
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            print("Mode=",phase)


            num_correct_batch_visibility = 0
            total_num_correct = 0
            total_correct_num_landmark = 0
            running_loss = 0.0
            running_corrects = 0
            k = 0
            TOTAL_DISTANCE = 0
            # Iterate over data.
            for data in dataloders[phase]:
                k += 1
                # get the inputs
                inputs, labels, vislabel, img_name = data
                nlabels = []
                # TODO
                for lab in labels:
                    nlabels.append(lab.tolist())
                labels = nlabels
                labels = list(map(list, (zip(*labels))))
                labels = torch.from_numpy(np.array(labels))
                vnlabel = []
                for vlab in vislabel:
                    vnlabel.append(vlab.tolist())
                vislabel = vnlabel
                vislabel = list(map(list, (zip(*vislabel))))
                vislabel = torch.from_numpy(np.array(vislabel))


                x = torch.FloatTensor([1])
                longTemp = torch.LongTensor([1])
                labels = labels.type_as(x)  # 将labels的类型转化为torch.FloatTensor
                labels = labels.cuda()
                vislabel = vislabel.type_as(x)
                vislabel = vislabel.cuda()
                if torch.cuda.is_available():
                    inputs = Variable(inputs).cuda()
                    labels = Variable(labels).cuda()
                    vislabel = Variable(vislabel).cuda()
                else:
                    inputs, labels, vislabel = Variable(inputs), Variable(labels), Variable(vislabel)



                # forward
                outputs = model(inputs)

                # print(outputs.shape)
                outputs = outputs.type_as(labels)

                criterionVis = nn.CrossEntropyLoss()
                criterionPos = nn.MSELoss()

                outputs_pos = outputs[:, 0:8]
                outputs_vis = outputs[:, 8:20]
                # vTr = torch.LongTensor([1])
                # vvv = vvv.type_as(vTr)
                # vislabel = vislabel.type_as(vTr)
                # print(type(vvv))
                out_vis_1 = outputs_vis[:, 0:3]
                out_vis_2 = outputs_vis[:, 3:6]
                out_vis_3 = outputs_vis[:, 6:9]
                out_vis_4 = outputs_vis[:, 9:12]

                label_vis1 = vislabel[:, 0:3]
                label_vis2 = vislabel[:, 3:6]
                label_vis3 = vislabel[:, 6:9]
                label_vis4 = vislabel[:, 9:12]

                _, pred_visibility1 = torch.max(out_vis_1, 1)
                # if pred_visibility1.item() == 2:
                for (idx,x) in enumerate(pred_visibility1):
                    if x == 2:
                        outputs_pos[idx,0] = outputs_pos[idx,1] = 0

                _, pred_visibility2 = torch.max(out_vis_2, 1)
                for (idx,x) in enumerate(pred_visibility2):
                    if x == 2:
                        outputs_pos[idx,2] = outputs_pos[idx,3] = 0  

                _, pred_visibility3 = torch.max(out_vis_3, 1)
                for (idx,x) in enumerate(pred_visibility3):
                    if x == 2:
                        outputs_pos[idx,4] = outputs_pos[idx,5] = 0  

                _, pred_visibility4 = torch.max(out_vis_4, 1)
                for (idx,x) in enumerate(pred_visibility4):
                    if x == 2:
                        outputs_pos[idx,6] = outputs_pos[idx,7] = 0

                _,label_visibility1 = torch.max(label_vis1, 1)


                _,label_visibility2 = torch.max(label_vis2, 1)


                _,label_visibility3 = torch.max(label_vis3, 1)


                _,label_visibility4 = torch.max(label_vis4, 1)

                num_correct_batch_visibility = (pred_visibility1 == label_visibility1).sum() +(pred_visibility2 == label_visibility2).sum()+(pred_visibility3 == label_visibility3).sum()+(pred_visibility4 == label_visibility4).sum()
                # print("num_correct_batch_visibility:",num_correct_batch_visibility)
                total_num_correct += int(num_correct_batch_visibility)
                
                
                ooo1 = outputs_pos.cpu().detach().numpy()
                labeln = labels.cpu().detach().numpy()
                # outputs_pos_ = outputs_pos.cpu().
                if k % 100 == 0:
                    count = 0
                    for (img, land, lanlab) in zip(img_name, ooo1,labeln):
                        count += 1
                        if count % 4 != 0:
                            continue
                        img = Image.open(img)
                        # plt.
                        plt.imshow(img, cmap=plt.gray())
                        plt.scatter(land[0], land[1],c='g')
                        plt.scatter(land[2], land[3],c='g')
                        plt.scatter(land[4], land[5],c='g')
                        plt.scatter(land[6], land[7],c='g')
                        plt.scatter(lanlab[0], lanlab[1],c='r')
                        plt.scatter(lanlab[2], lanlab[3],c='r')
                        plt.scatter(lanlab[4], lanlab[5],c='r')
                        plt.scatter(lanlab[6], lanlab[7],c='r')
                        # plt
                        plt.savefig("./outputImg/" + phase + "/" + str(epoch) + "-" + str(k) + "-" + str(count) + ".jpg")
                        plt.close()
                criterionV1 = nn.CrossEntropyLoss()
                lossvis1 = criterionV1(out_vis_1,torch.max(label_vis1,1)[1])#(n ,1)
                lossvis2 = criterionV1(out_vis_2,torch.max(label_vis2,1)[1])#(n ,1)
                lossvis3 = criterionV1(out_vis_3,torch.max(label_vis3,1)[1])#(n ,1)
                lossvis4 = criterionV1(out_vis_4,torch.max(label_vis4,1)[1])#(n ,1)

                lossV = lossvis1 + lossvis2+lossvis3+lossvis4
                lossP = criterionPos(input=outputs_pos,target=labels)  # lossP #+lossV #
                
                loss = lossP + lossV * (200.-epoch)/200.


                # zero the parameter gradients
                optimizer.zero_grad()

                if phase == train_filename:
                    loss.backward()
                    optimizer.step()

                if k % 100 == 0:
                    print(k, " loss:", loss.item() / batch_size,"   vloss:",lossV.item())
                
                running_loss += float(loss)

                pos1 = torch.sqrt(torch.sum((outputs_pos[:,0:1+1] - labels[:,0:1+1])**2,1))
                pos2 = torch.sqrt(torch.sum((outputs_pos[:,2:3+1] - labels[:,2:3+1])**2,1))
                pos3 = torch.sqrt(torch.sum((outputs_pos[:,4:5+1] - labels[:,4:5+1])**2,1))
                pos4 = torch.sqrt(torch.sum((outputs_pos[:,6:7+1] - labels[:,6:7+1])**2,1))
                TOTAL_DISTANCE += float(pos1.sum()) + float(pos2.sum()) + float(pos3.sum()) + float(pos4.sum()) 

 
                correct_num_pos1 = (pos1 < 35).sum() 
                correct_num_pos2 = (pos2 < 35).sum() 
                correct_num_pos3 = (pos3 < 35).sum() 
                correct_num_pos4 = (pos4 < 35).sum() 
 
                total_correct_num_landmark += correct_num_pos1.item() + correct_num_pos2.item() + correct_num_pos3.item() +correct_num_pos4.item()

            epoch_loss = running_loss / (dataset_sizes[phase]/batch_size)
            # epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} '.format(
                phase, epoch_loss/512))


                # print(landmark.size())
                # print(landmark[:,0:1+1].size())

                # print("*&*&"*10,torch.sum((landmark[:,0:1+1] - landmark_labels[:,0:1+1])**2,1))
                # break
            
            print("*"*20,'\n{} Total Loss: {:.4f} \nvisibility classify accurancy: {:.4f}   \nDPL acc:{:.4f} \nscore:{:.4f}'.format(phase, epoch_loss/512,total_num_correct/(dataset_sizes[phase]*4),
                                                                                                                                        total_correct_num_landmark/(dataset_sizes[phase]*4),TOTAL_DISTANCE/(dataset_sizes[phase]*4)/512),"\n","*"*20)
            # print(' landmark_point_DPL_acc:'+str((total_correct_num_landmark/(dataset_sizes[phase]*4))))
            if epoch % 2 == 0 and phase == train_filename:
                torch.save(model.state_dict(), './models/DeepFashion_'+str(epoch)+'.pth')
                # model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # torch.save(model.state_dict(), './models/DeepFashion.pth')
    # model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    data_transforms = {
        train_filename: transforms.Compose([
            # transforms.Resize((256, 256)),
            # transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        val_filename: transforms.Compose([
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        test_filename: transforms.Compose([
            # transforms.Resize((256, 256)),
            # transforms.CenterCrop(5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
    }
    image_datasets = {x: customData(img_path='./',
                                    txt_path=('./CSV/' + x + '.csv'),
                                    data_transforms=data_transforms,
                                    dataset=x) for x in
                      [train_filename, test_filename, val_filename]}
    dataset_sizes = {x: len(image_datasets[x]) for x in
                      [train_filename, test_filename, val_filename]}
    dataloders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                 batch_size=24,
                                                 shuffle=True,
                                                 num_workers=4) for x in
                      [train_filename, test_filename, val_filename]}

    vgg16 = Net(models.resnet18(pretrained=True))
    if torch.cuda.is_available():
        vgg16 = vgg16.cuda()

    criterion = nn.MSELoss()
    optimizer_ft = torch.optim.Adam(vgg16.parameters(), lr=1e-3)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.9)
    model_ft = train_model(model=vgg16,
                           criterion=criterion,
                           optimizer=optimizer_ft,
                           scheduler=exp_lr_scheduler,
                           batch_size=24,
                           num_epochs=200)
