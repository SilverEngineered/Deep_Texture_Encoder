import tensorflow as tf
from utils import nnUtils
import numpy as np
import copy
import cv2
from matplotlib import pyplot as plt

def residual_block(x,num_filters,kernel_size,name="resid_block"):
    shortcut = x
    x = tf.layers.batch_normalization(x)
    x = tf.nn.leaky_relu(x)
    x = tf.layers.conv2d(x,3,kernel_size, padding='same',activation=None)

    x = tf.layers.batch_normalization(x)
    x = tf.nn.leaky_relu(x)
    x = tf.layers.conv2d(x,3,kernel_size, padding='same',activation=None)
    x = x + shortcut
    return tf.nn.relu(x)
def small_network(x,reuse=False, name="encoder", activation=tf.nn.relu,num_filters=32, kernel_size=[5,5],stride=[1,1]):
    x = tf.layers.conv2d(x,3,kernel_size,stride, padding="SAME")
    x = tf.layers.conv2d(x,3,kernel_size,stride, padding="SAME")
    return x
def encoder(x,vector_dims,reuse=False, name="encoder", activation=tf.nn.relu,num_filters=32, kernel_size=[5,5],stride=[1,1]):
    with tf.variable_scope(name, reuse=reuse):
        x = residual_block(x,num_filters,kernel_size)
        x = residual_block(x,num_filters,kernel_size)
        x = residual_block(x,num_filters,kernel_size)
        x = residual_block(x,num_filters,kernel_size)
        x = tf.contrib.slim.conv2d_transpose(x, 1, kernel_size, stride, padding='SAME')
        x = tf.layers.batch_normalization(x)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x,vector_dims, activation=activation)
        return x
def decoder(x,image_shape,reuse=False, name="decoder", activation=tf.nn.relu,num_filters=128,stride=[2,2],kernel=[4,4]):
    with tf.variable_scope(name, reuse=reuse):
        x = tf.expand_dims(tf.expand_dims(x, 1), 1)
        print(x.shape)
        x = tf.layers.dense(x,int(image_shape[0]*image_shape[1]*image_shape[2]/16))
        print(x.shape)
        x = tf.reshape(x,[-1,int(image_shape[0]/4),int(image_shape[1]/4),image_shape[2]])
        print(x.shape)
        x = tf.layers.conv2d_transpose(x,num_filters,kernel,stride, padding="SAME")
        print(x.shape)
        x = tf.layers.conv2d_transpose(x,int(num_filters/2),kernel,stride, padding="SAME")
        print(x.shape)
        x = tf.layers.conv2d_transpose(x,int(num_filters/4),kernel,stride, padding="SAME")
        print(x.shape)
        x = tf.layers.conv2d_transpose(x,int(num_filters/8),kernel,stride, padding="SAME")
        print(x.shape)
        x = tf.layers.conv2d_transpose(x,3,kernel,stride, padding="SAME")
        print(x.shape)
        x = tf.layers.conv2d(x,3,kernel,stride, padding="SAME")
        print(x.shape)
        x = tf.layers.conv2d(x,3,kernel,stride, padding="SAME")
        print(x.shape)
        x = tf.layers.conv2d(x,3,kernel,stride, padding="SAME")
        print(x.shape)
        #exit()
        return x
def loss(masks,ground_truth,textured_images,num_textures, residual):
    loss = 0
    for i in range(num_textures):
        #print(masks[i])
        #print(textured_images[i])

        #Mask
        if residual:
            recreated_image = masks[i] + textured_images[i]
        else:
            recreated_image = masks[i]
        difference = recreated_image - ground_truth
        squared_difference = tf.square(difference)
        reduced_sum = tf.reduce_sum(squared_difference)
        if residual:
            loss += tf.reduce_sum(tf.square(tf.subtract(masks[i] + textured_images[i],ground_truth)))
        else:
            loss +=reduced_sum
    return loss
