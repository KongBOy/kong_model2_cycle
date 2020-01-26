from glob import glob
import numpy as np 
from imageio import imread
import scipy.misc
import time

from module_kong import build_CycleGAN

import tensorflow as tf

import cv2

def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0])
    img_B = imread(image_path[1])
    if not is_testing:
        img_A = scipy.misc.imresize(img_A, [load_size, load_size])
        img_B = scipy.misc.imresize(img_B, [load_size, load_size])
        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)
    else:
        img_A = scipy.misc.imresize(img_A, [fine_size, fine_size])
        img_B = scipy.misc.imresize(img_B, [fine_size, fine_size])

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB


def train():
    start_time = time.time()
    discriminator_a, discriminator_b, generator_a2b, generator_b2a, GAN_a, GAN_b = build_CycleGAN()
    print("create model cost time:",time.time() - start_time)
    epoch = 200
    epoch_step = 100
    counter = 1
    
    dataset_dir = "horse2zebra"
    # for epoch in range(args.epoch):
    for epoch in range(epoch):
        train_size  = 1e8
        batch_size  = 1
        # dataA = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainA'))
        # dataB = glob('./datasets/{}/*.*'.format(self.dataset_dir + '/trainB'))
        dataA = glob('./datasets/{}/*.*'.format(dataset_dir + '/trainA'))
        dataB = glob('./datasets/{}/*.*'.format(dataset_dir + '/trainB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)

        # batch_idxs = min(min(len(dataA), len(dataB)), args.train_size) // self.batch_size
        batch_idxs = min(min(len(dataA), len(dataB)), train_size) // batch_size
        # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)

        for idx in range(0, batch_idxs):
            #################################################################################################################
            # Load Batch data
            batch_files = list(zip(dataA[idx * batch_size : (idx + 1) * batch_size],
                                   dataB[idx * batch_size : (idx + 1) * batch_size])) ### [('./datasets/horse2zebra/trainA\\n02381460_1766.jpg', './datasets/horse2zebra/trainB\\n02391049_22.jpg')]
            batch_images = [load_train_data(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch_images).astype(np.float32) ### batch_images.shape (1, 256, 256, 6)
            
            ##################################################################################################################################################################################################################################
            # Update D network
            # discriminator_a.trainable = True   ### 把 discriminator 調成可訓練
            # discriminator_b.trainable = True   ### 把 discriminator 調成可訓練

            real_a = batch_images[:,:,:,0:3]
            real_b = batch_images[:,:,:,3:6]
            # print("real_a",real_a[0])
            # cv2.imshow("real_a",real_a[0])
            # cv2.waitKey(0)
            fake_b = generator_a2b(real_a) ### 丟進去要的形式要是 BHWC
            print(fake_b)
            fake_b_concat_real_b = tf.concat([fake_b, real_b]  ,axis=0)
            y1 = tf.constant( [[0.0]]*batch_size + [[1.0]]*batch_size )
            discriminator_b.train_on_batch( fake_b_concat_real_b, y1)
            #################################################################################################################
            # Update G network and record fake outputs
            # discriminator_a.trainable = False   ### 把 discriminator 調成可訓練
            # discriminator_b.trainable = False   ### 把 discriminator 調成可訓練
            
            #################################################################################################################
            ##################################################################################################################################################################################################################################
            # counter += 1 原始寫這邊，我把它調到下面去囉
            cost_time = time.time() - start_time
            hour = cost_time//3600 ; minute = cost_time%3600//60 ; second = cost_time%3600%60
            print(("Epoch: [%2d] [%4d/%4d] time: %4.4f, %2d:%02d:%02d counter:%d" % (
                epoch, idx, batch_idxs, time.time() - start_time,hour, minute, second, counter)))

            counter += 1  ### 調到這裡
train()