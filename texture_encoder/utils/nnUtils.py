import tensorflow as tf
import sys
import matplotlib.pyplot as plt
import time
import random
import numpy as np
from tqdm import tqdm
from utils import data_utilsWDB as utils
import os
import cv2


def import_images(data_path,color_dim=3):
    src=data_path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    X_train=[]
    for each in os.listdir(src):
        path=os.path.join(src,each)
        img = cv2.imread(path, 1)
        if color_dim==1:
            img=np.expand_dims(img,axis=2)
        X_train.append(img)
    return utils.scale_data(np.asarray(X_train))
def import_images_ignore(data_path,ignore_path, size = [2500,256,256,3]):
    src=data_path
    dir_path = os.path.dirname(os.path.realpath(__file__))
    num_textures=len(os.listdir(src)) - 1
    all_images=np.zeros([num_textures,size[0],size[1],size[2],size[3]])
    count = 0
    for each in tqdm(os.listdir(src)):
        if each != ignore_path:
            path=os.path.join(src,each)
            all_images[count] = import_images(path)
            count +=1
    return all_images
