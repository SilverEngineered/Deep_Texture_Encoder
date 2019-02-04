import tensorflow as tf
import numpy as np
import random
from modules import*
from utils import nnUtils
from utils import data_utilsWDB as util
import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm
import datetime
import os
class phi(object):
    def __init__(self, sess, args):
        self.sess = sess
        self.dataset_dir = args.dataset_dir
        self.generalized_data = args.generalized_data
        self.lr = args.lr
        self.epoch = args.epoch
        self.image_shape = args.image_shape
        self.vector_dims = args.vector_dims
        self.small_network = args.small_network
        self.residual = args.residual
        self.load_checkpoint=args.load
        self.num_textures=args.num_textures
        self.num_images = args.num_images
        if not self.load_checkpoint:
            self.texture_data = nnUtils.import_images_ignore(args.dataset_dir,args.generalized_data,size=[args.num_images,args.image_shape[0],args.image_shape[1],args.image_shape[2]])
            self.real_data = nnUtils.import_images(os.path.join(args.dataset_dir,args.generalized_data))
            print("Imported Data!")
            self.num_textures = self.texture_data.shape[0]

        if not self.small_network:
            self.encoder = encoder
            self.decoder = decoder
        else:
            self.encoder = small_network
        self.build()
        if self.load_checkpoint:
            self.load()
    def build(self):
        self.x = tf.placeholder(tf.float32, [None, self.image_shape[0], self.image_shape[1], 3])
        self.y = tf.placeholder(tf.float32, [self.image_shape[0], self.image_shape[1], 3])
        if not self.small_network:
            self.encoder_net = self.encoder(self.x,self.vector_dims, reuse=False)
            self.decoder_net = self.decoder(self.encoder_net,self.image_shape,reuse=False)
        else:
            self.encoder_net = self.encoder(self.x,reuse=False)
        self.vars=tf.trainable_variables()
        if not self.small_network:
            self.loss=loss(self.decoder_net,self.y,self.x,self.num_textures,self.residual)
        else:
            self.loss=loss(self.encoder_net,self.y,self.x,self.num_textures,self.residual)
        self.optim = tf.train.AdamOptimizer(self.lr).minimize(self.loss, var_list=self.vars)
        self.saver = tf.train.Saver()
        print("Built!")

    def train(self,args):
        '''
        Trains the triplet network by having it learn an embedding into args.vector_dims dimensional space.
        :param args: StringArray, Arguments passed in from argument parser
        '''

        loss_scalar = tf.summary.scalar("Loss", self.loss)
        init=tf.global_variables_initializer()
        train_writer = tf.summary.FileWriter( './logs/train ', self.sess.graph)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        checkpoint = os.path.join(dir_path,"checkpoints/",)

        self.sess.run(init)
        iter=0
        print("Training initialized")
        all_loss=[]
        for i in range(self.epoch):
            index_transform = random.randint(0, self.real_data.shape[0]-1)
            texture_batch=np.zeros([self.num_textures,self.image_shape[0],self.image_shape[1],self.image_shape[2]])
            for x in range(self.num_textures):
                texture_batch[x]=self.texture_data[x][index_transform]
            real_batch=self.real_data[index_transform]
            if not self.small_network:
                loss_data, _, generated, _ = self.sess.run([self.loss,self.optim,self.decoder_net, self.encoder_net],feed_dict={self.x: texture_batch, self.y: real_batch})
            else:
                loss_data, _, generated = self.sess.run([self.loss,self.optim,self.encoder_net],feed_dict={self.x: texture_batch, self.y: real_batch})
            #print(loss_data)
            img_texture = 5
            img_texture_2 = 6
            img_texture_3 = 7
            img_mod = 100
            img_path = './results/'
            if i%img_mod==0:
                if self.residual:
                    generated_image = util.scale_data(generated[img_texture] + texture_batch[img_texture],[0,255])
                    cv2.imwrite(img_path + 'generated_images/' + str(int(i/img_mod)) + ".jpg", generated_image)
                    generated_image = util.scale_data(generated[img_texture_2] + texture_batch[img_texture_2],[0,255])
                    cv2.imwrite(img_path + 'generated_images_2/' + str(int(i/img_mod)) + ".jpg", generated_image)
                    generated_image = util.scale_data(generated[img_texture_3] + texture_batch[img_texture_3],[0,255])
                    cv2.imwrite(img_path + 'generated_images_3/' + str(int(i/img_mod)) + ".jpg", generated_image)
                    mask = util.scale_data(generated[img_texture],[0,255])
                    cv2.imwrite(img_path + 'masks/' + str(int(i/img_mod)) + ".jpg", mask)
                    mask = util.scale_data(generated[img_texture_2],[0,255])
                    cv2.imwrite(img_path + 'masks_2/' + str(int(i/img_mod)) + ".jpg", mask)
                    mask = util.scale_data(generated[img_texture_3],[0,255])
                    cv2.imwrite(img_path + 'masks_3/' + str(int(i/img_mod)) + ".jpg", mask)
                else:
                    img = util.scale_data(generated[img_texture],[0,255])
                    cv2.imwrite(img_path + 'generated_images/' + str(int(i/img_mod)) + ".jpg", img)
                    img = util.scale_data(generated[img_texture_2],[0,255])
                    cv2.imwrite(img_path + 'generated_images_2/' + str(int(i/img_mod)) + ".jpg", img)
                    img = util.scale_data(generated[img_texture_3],[0,255])
                    cv2.imwrite(img_path + 'generated_images_3/' + str(int(i/img_mod)) + ".jpg", img)
                target_image = util.scale_data(real_batch,[0,255])
                cv2.imwrite(img_path + 'target/' + str(int(i/img_mod)) + ".jpg", target_image)
                original_image = util.scale_data(texture_batch[img_texture],[0,255])
                cv2.imwrite(img_path + 'original/' + str(int(i/img_mod)) + ".jpg", original_image)
                original_image = util.scale_data(texture_batch[img_texture_2],[0,255])
                cv2.imwrite(img_path + 'original_2/' + str(int(i/img_mod)) + ".jpg", original_image)
                original_image = util.scale_data(texture_batch[img_texture_3],[0,255])
                cv2.imwrite(img_path + 'original_3/' + str(int(i/img_mod)) + ".jpg", original_image)
                self.saver.save(self.sess, checkpoint, global_step=img_mod)

                print("Loss: " + str(loss_data))
    def load(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        checkpoint = os.path.join(dir_path,"checkpoints/",)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint))
    def transform_image(self,img):
        _, generated = self.sess.run([self.encoder_net,self.decoder_net],feed_dict={self.x: img})
        return generated[0]
