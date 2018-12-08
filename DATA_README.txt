=====================================================================
Large-scale Fashion Recognition and Retrieval (DeepFashion) Dataset
=====================================================================

======================================
Fashion Landmark Detection Benchmark
======================================

--------------------------------------------------------
By Multimedia Lab, The Chinese University of Hong Kong
--------------------------------------------------------

For more information about the dataset, visit the project website:

  http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

If you use the dataset in a publication, please cite the papers below:

  @inproceedings{liu2016deepfashion,
 	author = {Ziwei Liu, Ping Luo, Shi Qiu, Xiaogang Wang, and Xiaoou Tang},
 	title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},
 	booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 	month = June,
 	year = {2016} 
  }

  @inproceedings{liu2016fashionlandmark,
 	author = {Ziwei Liu, Sijie Yan, Ping Luo, Xiaogang Wang, and Xiaoou Tang},
 	title = {Fashion Landmark Detection in the Wild},
 	booktitle = {European Conference on Computer Vision (ECCV)},
 	month = October,
 	year = {2016} 
  }

Please note that we do not own the copyrights to these images. Their use is RESTRICTED to non-commercial research and educational purposes.



========================
Change Log
========================

Version 1.0, released on 25/07/2016
Version 1.1, released on 31/01/2018, add human joint annotations



========================
File Information
========================

- Images (Img/img.zip)
    123,016 diverse clothes images. See IMAGE section below for more info.

- Bounding Box Annotations (Anno/list_bbox.txt)
    bounding box labels. See BBOX LABELS section below for more info.

- Fashion Landmark Annotations (Anno/list_landmarks.txt)
	fashion landmark labels. See LANDMARK LABELS section below for more info.

- Human Joint Annotations (Anno/list_joints.txt)
	human joint labels. See JOINT LABELS section below for more info.

- Evaluation Partitions (Eval/list_eval_partition.txt)
	image names for training, validation and testing set respectively. See EVALUATION PARTITIONS section below for more info.



=========================
IMAGE
=========================

------------ img.zip ------------

format: JPG

---------------------------------------------------

Notes:
1. The long side of images are resized to 512;
2. The aspect ratios of original images are kept unchanged.

---------------------------------------------------



=========================
BBOX LABELS
=========================

------------ list_bbox.txt ------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image name> <bbox location>

---------------------------------------------------

Notes:
1. The order of bbox labels accords with the order of entry names;
2. In bbox location, "x_1" and "y_1" represent the upper left point coordinate of bounding box, "x_2" and "y_2" represent the lower right point coordinate of bounding box. Bounding box locations are listed in the order of [x_1, y_1, x_2, y_2].

---------------------------------------------------



=========================
LANDMARK LABELS
=========================

--------------- list_landmarks.txt --------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image name> <clothes type> <variation type> [<landmark visibility 1> <landmark location x_1> <landmark location y_1>, ... <landmark visibility 8> <landmark location x_8> <landmark location y_8>]

---------------------------------------------------

Notes:
1. The order of landmark labels accords with the order of entry names;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes. Upper-body clothes possess six fashion landmarks, lower-body clothes possess four fashion landmarks, full-body clothes possess eight fashion landmarks;
3. In variation type, "1" represents normal pose, "2" represents medium pose, "3" represents large pose, "4" represents medium zoom-in, "5" represents large zoom-in;
4. In landmark visibility state, "0" represents visible, "1" represents invisible/occluded, "2" represents truncated/cut-off;
5. For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left hem", "right hem"]; For lower-body clothes, landmark annotations are listed in the order of ["left waistline", "right waistline", "left hem", "right hem"]; For upper-body clothes, landmark annotations are listed in the order of ["left collar", "right collar", "left sleeve", "right sleeve", "left waistline", "right waistline", "left hem", "right hem"].

---------------------------------------------------



=========================
JOINT LABELS
=========================

--------------- list_joints.txt --------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image name> <clothes type> <variation type> [<joint visibility 1> <joint location x_1> <joint location y_1>, ... <joint visibility 14> <joint location x_14> <joint location y_14>]

---------------------------------------------------

Notes:
1. The order of joint labels accords with the order of entry names. Overall there are fourteen human joints;
2. In clothes type, "1" represents upper-body clothes, "2" represents lower-body clothes, "3" represents full-body clothes;
3. In variation type, "1" represents normal pose, "2" represents medium pose, "3" represents large pose, "4" represents medium zoom-in, "5" represents large zoom-in;
4. In landmark visibility state, "0" represents visible, "1" represents invisible.

---------------------------------------------------



=========================
EVALUATION PARTITIONS
=========================

------------- list_eval_partition.txt -------------

First Row: number of images
Second Row: entry names

Rest of the Rows: <image name> <evaluation status>

---------------------------------------------------

Notes:
1. In evaluation status, "train" represents training image, "val" represents validation image, "test" represents testing image;
2. Please refer to the paper "Fashion Landmark Detection in the Wild" for more details.

---------------------------------------------------



=========================
Contact
=========================

Please contact Ziwei Liu (zwliu.hust@gmail.com) for questions about the dataset.