from __future__ import print_function, division
from torch.utils.data import Dataset
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
import matplotlib.pyplot as plt
import numpy as np
import csv
from PIL import Image

def ChangSizeAndlandmark(file_box,file_landmark,saveCSVfilename,offset_imgpath):
	#制作字典及其对应的bbox坐标
	dict = {}
	bbox = open(file_box)
	k =0
	# print(bbox.readlines())
	for line in bbox:
		if k==0 or k == 1:#skip the first line and the second line
			k += 1
			continue
		line = line.strip().split()
		# print("line",line)
		# print(line)
		list1=[]
		list1.append(int(line[3]))
		list1.append(int(line[4]))
		list1.append(int(line[5]))
		list1.append(int(line[6]))
		# list1.append(int(line[7]))
		# list1.append(int(line[8]))
		dict[line[0]] = list1
		k += 1
	# print(dict)
	out = open(saveCSVfilename,'a', newline='')
	csv_write = csv.writer(out,dialect='excel')
	with open(file_landmark) as csvfile:
		csv_reader = csv.reader(csvfile)
		k = 0
		long = 2
		for row in csv_reader:
			k += 1
			img_path = offset_imgpath + row[0][2:]
			print("="*10,">>"," processing No: ",k," ",img_path)
			if img_path == "/home/hk/Desktop/Consumer-to-shop_Clothes_Retrieval_Benchmark/imgzip/img/TROUSERS/Pants/id_00000023/shop_01.jpg":
				continue
			img = Image.open(img_path)
			#裁剪图片
			list_box = dict[row[0][2:]]
			# print(list_box)
			crop_img = img.crop((list_box[0],list_box[1],list_box[2],list_box[3]))
			
			img_array = np.array(crop_img)
			try:
				h,w,d = img_array.shape
			except:
				continue
			maxinum = max(w,h)
			img_array = Image.fromarray(img_array)
			rate = 0.

			if w>h:
				long = 0
				rate = 224/w
				new_w = 224
				new_h = int(h*rate+0.5)
			else:
				long = 1
				rate = 224/h
				new_h = 224
				new_w = int(w*rate+0.5)
			
			img_resize = img_array.resize((new_w,new_h))

			x = np.array(img_resize)
			# print("crop_img.size()",x.shape)

			# img_pad = 0
			if long == 0:
				img_array = np.array(img_resize)
				up_pad = int((224 - new_h)/2+0.5)
				down_pad = (224-new_h) - up_pad 
				# print("new_w",new_w,"new_h",new_h,"w:",w,"h:",h,"up_pad:",up_pad,"down_pad:",down_pad)
				# img = np.pad(img_array,((up_pad,down_pad),(0,0)))

				channel_one = img_array[:,:,0]
				channel_two = img_array[:,:,1]
				channel_three = img_array[:,:,2]
				# print("channel_one.shape:",channel_one.shape)
				channel_one = np.pad(channel_one, ((up_pad, down_pad),(0, 0) ),  'constant', constant_values=(0,0))
				channel_two = np.pad(channel_two, ((up_pad, down_pad),(0, 0) ),  'constant', constant_values=(0,0))
				channel_three = np.pad(channel_three, ((up_pad, down_pad),(0, 0) ),  'constant', constant_values=(0,0))
				img_pad = np.dstack((channel_one,channel_two,channel_three))
					
				# new_x1 = x*rate
				# new_y2 = y*rate + up_pad
				# print("w>h:","int(row[4]):",int(row[4]),"int(list_box[0]):",int(list_box[0]))
				new_x1 = max( (int(row[4])-int(list_box[0]))*rate+0.5,0)
				new_y1 = max( (int(row[5])-int(list_box[1]))*rate+0.5 + up_pad,0)
				new_x2 = max( (int(row[7])-int(list_box[0]))*rate+0.5,0)
				new_y2 = max( (int(row[8])-int(list_box[1]))*rate+0.5 + up_pad,0)						
				new_x3 = max( (int(row[10])-int(list_box[0]))*rate+0.5,0)
				new_y3 = max( (int(row[11])-int(list_box[1]))*rate+0.5 + up_pad,0)
				new_x4 = max( (int(row[13])-int(list_box[0]))*rate+0.5,0)
				new_y4 = max( (int(row[14])-int(list_box[1]))*rate+0.5 + up_pad,0)
				new_x5 = max( (int(row[16])-int(list_box[0]))*rate+0.5,0)
				new_y5 = max( (int(row[17])-int(list_box[1]))*rate+0.5 + up_pad,0)
				new_x6 = max( (int(row[19])-int(list_box[0]))*rate+0.5,0)
				new_y6 = max( (int(row[20])-int(list_box[1]))*rate+0.5 + up_pad,0)		

				new_x7 = max( (int(row[22])-int(list_box[0]))*rate+0.5,0)
				new_y7 = max( (int(row[23])-int(list_box[1]))*rate+0.5 + up_pad,0)
				new_x8 = max( (int(row[25])-int(list_box[0]))*rate+0.5,0)
				new_y8 = max( (int(row[26])-int(list_box[1]))*rate+0.5 + up_pad,0)						
				if int(row[4])==0 and int(row[5]) ==0:
					new_x1 = 0
					new_y1 = 0
				if int(row[7])==0 and int(row[8]) ==0:
					new_x2 = 0
					new_y2 = 0					
				if int(row[10])==0 and int(row[11]) ==0:
					new_x3 = 0
					new_y3 = 0					
				if int(row[13])==0 and int(row[14]) ==0:
					new_x4 = 0
					new_y4 = 0
				if int(row[16])==0 and int(row[17]) ==0:
					new_x5 = 0
					new_y5 = 0					
				if int(row[19])==0 and int(row[20]) ==0:
					new_x6 = 0
					new_y6 = 0				
				if int(row[22])==0 and int(row[23]) ==0:
					new_x7 = 0
					new_y7 = 0
				if int(row[25])==0 and int(row[26]) ==0:
					new_x8 = 0
					new_y8 = 0						
				row[4] = int(new_x1)
				row[5] = int(new_y1)

				row[7] = int(new_x2)
				row[8] = int(new_y2)

				row[10] = int(new_x3)
				row[11] = int(new_y3)

				row[13] = int(new_x4)
				row[14] = int(new_y4)

				row[16] = int(new_x5)
				row[17] = int(new_y5)

				row[19] = int(new_x6)
				row[20] = int(new_y6)

				row[22] = int(new_x5)
				row[23] = int(new_y5)

				row[25] = int(new_x6)
				row[26] = int(new_y6)
			else:#long =1 h>w

				img_array = np.array(img_resize)
				left_pad = int((224 - new_w)/2+0.5)
				right_pad = int(224-new_w) - left_pad 
				# print("new_w",new_w,"new_h",new_h,"w:",w,"h:",h,"left_pad:",left_pad,"right_pad:",right_pad)
				channel_one = img_array[:,:,0]
				channel_two = img_array[:,:,1]
				channel_three = img_array[:,:,2]

				channel_one = np.pad(channel_one, ((0, 0),(left_pad, right_pad) ),  'constant', constant_values=(0,0))
				channel_two = np.pad(channel_two, ((0, 0),(left_pad, right_pad) ),  'constant', constant_values=(0,0))
				channel_three = np.pad(channel_three, ((0, 0),(left_pad, right_pad) ),  'constant', constant_values=(0,0))
				img_pad = np.dstack((channel_one,channel_two,channel_three))

				new_x1 = max( (int(row[4])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y1 = max( (int(row[5])-int(list_box[1]))*rate+0.5 ,0)
				new_x2 = max( (int(row[7])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y2 = max( (int(row[8])-int(list_box[1]))*rate+0.5 ,0)						
				new_x3 = max( (int(row[10])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y3 = max( (int(row[11])-int(list_box[1]))*rate+0.5 ,0)
				new_x4 = max( (int(row[13])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y4 = max( (int(row[14])-int(list_box[1]))*rate+0.5 ,0)
				new_x5 = max( (int(row[16])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y5 = max( (int(row[17])-int(list_box[1]))*rate+0.5 ,0)
				new_x6 = max( (int(row[19])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y6 = max( (int(row[20])-int(list_box[1]))*rate+0.5 ,0)		

				new_x7 = max( (int(row[22])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y7 = max( (int(row[23])-int(list_box[1]))*rate+0.5 ,0)
				new_x8 = max( (int(row[25])-int(list_box[0]))*rate+0.5+ left_pad,0)
				new_y8 = max( (int(row[26])-int(list_box[1]))*rate+0.5 ,0)						
				if int(row[4])==0 and int(row[5]) ==0:
					new_x1 = 0
					new_y1 = 0
				if int(row[7])==0 and int(row[8]) ==0:
					new_x2 = 0
					new_y2 = 0					
				if int(row[10])==0 and int(row[11]) ==0:
					new_x3 = 0
					new_y3 = 0					
				if int(row[13])==0 and int(row[14]) ==0:
					new_x4 = 0
					new_y4 = 0
				if int(row[16])==0 and int(row[17]) ==0:
					new_x5 = 0
					new_y5 = 0					
				if int(row[19])==0 and int(row[20]) ==0:
					new_x6 = 0
					new_y6 = 0				
				if int(row[22])==0 and int(row[23]) ==0:
					new_x7 = 0
					new_y7 = 0
				if int(row[25])==0 and int(row[26]) ==0:
					new_x8 = 0
					new_y8 = 0						
				row[4] = int(new_x1)
				row[5] = int(new_y1)

				row[7] = int(new_x2)
				row[8] = int(new_y2)

				row[10] = int(new_x3)
				row[11] = int(new_y3)

				row[13] = int(new_x4)
				row[14] = int(new_y4)

				row[16] = int(new_x5)
				row[17] = int(new_y5)

				row[19] = int(new_x6)
				row[20] = int(new_y6)

				row[22] = int(new_x5)
				row[23] = int(new_y5)

				row[25] = int(new_x6)
				row[26] = int(new_y6)




				#6 points
				# new_x1 = max( (int(row[4])-int(list_box[0]))*rate+0.5+ left_pad,0)
				# new_y1 = max( (int(row[5])-int(list_box[1]))*rate+0.5 ,0)
				# new_x2 = max( (int(row[7])-int(list_box[0]))*rate+0.5+ left_pad,0)
				# new_y2 = max( (int(row[8])-int(list_box[1]))*rate+0.5 ,0)						
				# new_x3 = max( (int(row[10])-int(list_box[0]))*rate+0.5+ left_pad,0)
				# new_y3 = max( (int(row[11])-int(list_box[1]))*rate+0.5 ,0)
				# new_x4 = max( (int(row[13])-int(list_box[0]))*rate+0.5+ left_pad,0)
				# new_y4 = max( (int(row[14])-int(list_box[1]))*rate+0.5 ,0)
				# new_x5 = max( (int(row[16])-int(list_box[0]))*rate+0.5+ left_pad,0)
				# new_y5 = max( (int(row[17])-int(list_box[1]))*rate+0.5 ,0)
				# new_x6 = max( (int(row[19])-int(list_box[0]))*rate+0.5+ left_pad,0)
				# new_y6 = max( (int(row[20])-int(list_box[1]))*rate+0.5 ,0)				
				# if int(row[4])==0 and int(row[5]) ==0:
				# 	new_x1 = 0
				# 	new_y1 = 0
				# if int(row[7])==0 and int(row[8]) ==0:
				# 	new_x2 = 0
				# 	new_y2 = 0					
				# if int(row[10])==0 and int(row[11]) ==0:
				# 	new_x3 = 0
				# 	new_y3 = 0					
				# if int(row[13])==0 and int(row[14]) ==0:
				# 	new_x4 = 0
				# 	new_y4 = 0
				# if int(row[16])==0 and int(row[17]) ==0:
				# 	new_x5 = 0
				# 	new_y5 = 0					
				# if int(row[19])==0 and int(row[20]) ==0:
				# 	new_x6 = 0
				# 	new_y6 = 0				
				# row[4] = int(new_x1)
				# row[5] = int(new_y1)

				# row[7] = int(new_x2)
				# row[8] = int(new_y2)

				# row[10] = int(new_x3)
				# row[11] = int(new_y3)

				# row[13] = int(new_x4)
				# row[14] = int(new_y4)

				# row[16] = int(new_x5)
				# row[17] = int(new_y5)

				# row[19] = int(new_x6)
				# row[20] = int(new_y6)


				#4 points
				# new_x1 = max( (int(row[4])-int(list_box[0]) )*rate + left_pad , 0)
				# new_y1 = max( (int(row[5])-int(list_box[1]) )*rate+0.5,1)
				# new_x2 = max( (int(row[7])-int(list_box[0]) )*rate + left_pad , 0)
				# new_y2 = max( (int(row[8])-int(list_box[1]) )*rate+0.5,1)						
				# new_x3 = max( (int(row[10])-int(list_box[0]) )*rate + left_pad , 0)
				# new_y3 = max( (int(row[11])-int(list_box[1]) )*rate+0.5,1)
				# new_x4 = max( (int(row[13])-int(list_box[0]) )*rate + left_pad , 0)
				# new_y4 = max( (int(row[14])-int(list_box[1]) )*rate+0.5,1)				
				# if int(row[4])==0 and int(row[5]) ==0:
				# 	new_x1 = 0
				# 	new_y1 = 0
				# if int(row[7])==0 and int(row[8]) ==0:
				# 	new_x2 = 0
				# 	new_y2 = 0					
				# if int(row[10])==0 and int(row[11]) ==0:
				# 	new_x3 = 0
				# 	new_y3 = 0					
				# if int(row[13])==0 and int(row[14]) ==0:
				# 	new_x4 = 0
				# 	new_y4 = 0

				# row[4] = int(new_x1)
				# row[5] = int(new_y1)

				# row[7] = int(new_x2)
				# row[8] = int(new_y2)

				# row[10] = int(new_x3)
				# row[11] = int(new_y3)

				# row[13] = int(new_x4)
				# row[14] = int(new_y4)
			
			Image.fromarray(img_pad).save(img_path)

			# plt.imshow(img_pad,cmap=plt.gray())
			# plt.scatter(new_x1, new_y1,c='r')
			# plt.scatter(new_x2, new_y2,c='r')
			# plt.scatter(new_x3, new_y3,c='r')
			# plt.scatter(new_x4, new_y4,c='r')
			# plt.scatter(new_x5, new_y5,c='r')
			# plt.scatter(new_x6, new_y6,c='r')			
			# plt.scatter(new_x7, new_y7,c='r')
			# plt.scatter(new_x8, new_y8,c='r')	
			# plt.show()
			# if k > 10:
			# 	break
			csv_write.writerow(row)  

	csvfile.close()
	out.close()

   
if __name__ == '__main__':
	offset_imgpath = "/home/hk/Desktop/Consumer-to-shop_Clothes_Retrieval_Benchmark/imgzip/newimg/"
	bbox_file = "/home/hk/Desktop/Consumer-to-shop_Clothes_Retrieval_Benchmark/Anno/list_bbox_consumer2shop.txt"
	train4_file = "/home/hk/Desktop/Consumer-to-shop_Clothes_Retrieval_Benchmark/all_expand_len8/test_expandto8.csv"
	save_new_landmark_file = "/home/hk/Desktop/Consumer-to-shop_Clothes_Retrieval_Benchmark/all_expand_len8/test_expandto8_bbox_pad.csv"
	ChangSizeAndlandmark(bbox_file,train4_file,save_new_landmark_file,offset_imgpath)
