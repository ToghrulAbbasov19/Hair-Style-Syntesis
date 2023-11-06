import torch
from torchvision import transforms
from torchvision import datasets
import pandas as pd
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F
import copy
import math
from StarGanV2 import Generator, StyleEncoder
from BiSeNet import BiSeNet
import glob
from Dataset import transform

gen=Generator()
stylenc=StyleEncoder()
bisenet=BiSeNet(19)  #Load pretrained models

root=""  #Root of Dataset
# print(os.listdir(root))

def vis_parsing_maps(im, parsing_anno, stride, save_im=False, save_path=None):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.zeros((256, 256))
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]
    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0, vis_parsing_anno_color, 1, 0) 
    return vis_im



def parsing(image_path=None, net=bisenet):

    # if not os.path.exists(respth):
    #     os.makedirs(respth)

    # n_classes = 19
    # net = BiSeNet(n_classes=n_classes)
    # net.cuda()
    # save_pth = osp.join('./pretrained_network/parsing', cp)
    # net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # with torch.no_grad():

    #     for image_path in os.listdir(dspth):
    img = Image.open(image_path)
    image = img.resize((256, 256), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    out = net(img)[0]
    parsing = out.squeeze(0).cpu().detach().numpy().argmax(0) 
    parsing = parsing + 100
    parsing[parsing == 100] = 0
    parsing[parsing == 101] = 1
    parsing[parsing == 102] = 6
    parsing[parsing == 103] = 7
    parsing[parsing == 104] = 4
    parsing[parsing == 105] = 5
    parsing[parsing == 106] = 3
    parsing[parsing == 107] = 8
    parsing[parsing == 108] = 9
    parsing[parsing == 109] = 15
    parsing[parsing == 110] = 2
    parsing[parsing == 111] = 10
    parsing[parsing == 112] = 11
    parsing[parsing == 113] = 12
    parsing[parsing == 114] = 17
    parsing[parsing == 115] = 16
    parsing[parsing == 116] = 18
    parsing[parsing == 117] = 13
    parsing[parsing == 118] = 14
               
    return vis_parsing_maps(image, parsing, stride=1, save_im=False, save_path=None)




def image_resizer(img):
  img = np.array(img)
  assert img.shape == (256, 256, 3)
  
  img -= img.min()
  img /= img.max()  
  img *= 255

  # img = (img+1)*127.5
  img = img.astype(np.uint8)
  img = img.clip(0, 255)
  return img



image=os.path.join(root, "20.jpg")
reference=os.path.join(root, "60.jpg")


image=Image.open(image).convert("RGB")
image=transform(image)
reference=Image.open(reference).convert("RGB")
reference=transform(reference)


image=image.unsqueeze(0)
reference=reference.unsqueeze(0)
y=torch.tensor([0]) #This depends on gender

style=stylenc(reference, y)

gen_image=gen(image, style)

gen_image=gen_image[0]


org_img=np.transpose(image[0].to("cpu").detach().numpy(), [1,2,0])
gen_img=np.transpose(gen_image.to("cpu").detach().numpy(), [1,2,0])
ref_img=np.transpose(reference[0].to("cpu").detach().numpy(), [1,2,0])


gen_img_=image_resizer(gen_img)
gen_img_=Image.fromarray(gen_img_)

gen_img_.save("E:\\newpic.png")
gen_img_=parsing("E:\\newpic.png")

copy_img=gen_img.copy()
for i in range(256):
  for j in range(256):
      # if int(gen_img_[i, j, 0])==153 and int(gen_img_[i, j, 1])==51 and int(gen_img_[i, j, 2])==0:
      # if list(gen_img_[i, j])==[153, 51, 0] or list(gen_img_[i, j])==[102, 0, 153] or list(gen_img_[i, j])==[0, 153, 0] or list(gen_img_[i, j])==[153, 102, 0]:
      if list(gen_img_[i,j])!=[0, 85, 255]:
        copy_img[i, j, 0] = org_img[i, j, 0]
        copy_img[i, j, 1] = org_img[i, j, 1]
        copy_img[i, j, 2] = org_img[i, j, 2]

plt.subplot(1,4,1)
plt.imshow(org_img)

plt.subplot(1,4,2)
plt.imshow(ref_img)

plt.subplot(1,4,3)
plt.imshow(gen_img)

plt.subplot(1,4,4)
plt.imshow(copy_img)

plt.show()  # This will plot original image, reference image, output of stargan, and original image with new hairstyle
