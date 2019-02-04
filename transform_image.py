import tensorflow as tf
import os
import argparse
from models.phi import phi
import numpy as np
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_dir', dest='dataset_dir', default='data', help='path of the directories containing textured images')
parser.add_argument('--generalized_data', dest='generalized_data', default='real', help='path within dataset_dir that contains the images to be generalized to')
parser.add_argument('--test_dir_x', dest='test_dir_x', default='cropped_10k_lsun_test', help='path of the test dataset')
parser.add_argument('--ck_path', dest='ck_path', default='checkpoints', help='checkpoint path')
parser.add_argument('--epoch', dest='epoch', type=int, default=200000, help='# of epoch')
parser.add_argument('--learning_rate', dest='lr', type=float, default=.0001,help='learning rate')
parser.add_argument('--image_shape', dest='image_shape',default=[256,256,3], help='shape of each image')
parser.add_argument('--vector_dims', dest='vector_dims',default=1000, help='Dimensionality of the encoded vector')
parser.add_argument('--small_network', dest='small_network',default=False, help='use simple network')
parser.add_argument('--residual', dest='residual',default=False, help='use a residual network')
parser.add_argument('--load', dest='load',default=True, help='Load weights')
args = parser.parse_args()


tfconfig = tf.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
with tf.Session(config=tfconfig) as sess:
    model = phi(sess, args)
    img = model.transform_image(np.random.rand(1,256,256,3))
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    exit()
