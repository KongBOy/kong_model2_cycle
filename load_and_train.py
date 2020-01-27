from glob import glob
import numpy as np 
# from imageio import imread
from scipy.misc import imread
import scipy.misc
import time

from module_kong import build_CycleGAN

import tensorflow as tf

import cv2



def load_train_data(image_path, load_size=286, fine_size=256, is_testing=False):
    img_A = imread(image_path[0], mode="RGB")
    img_B = imread(image_path[1], mode="RGB")
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


def sample_test_data(src_path, dst_path, generator_a2b,counter = 0, epoch = 0):
    test_img = imread(src_path)
    test_img = test_img/127.5 - 1.
    test_img = test_img.reshape( 1, test_img.shape[0], test_img.shape[1], test_img.shape[2] )
    test_img = test_img.astype(np.float32)
    result = generator_a2b(test_img)
    result = result.numpy()
    result = result.reshape(result.shape[1],result.shape[2],result.shape[3])
    result = (result+1.) /2.
    scipy.misc.imsave("%s-epoch%04i-%04i.jpg"%(dst_path,epoch,counter), result)

def train():
    start_time = time.time()
    discriminator_a, discriminator_b, generator_a2b, generator_b2a, GAN_b2a, GAN_a2b = build_CycleGAN()
    print("create model cost time:",time.time() - start_time)
    epochs = 200
    epoch_step = 100
    counter = 1
    
    dataset_dir = "horse2zebra"
    y_d = tf.constant(  [ [[ [0.0] ]*16]*16  ]*1 + [ [[ [1.0] ]*16]*16  ]*1 , dtype=tf.float32)
    y_g = tf.constant(  [ [[ [1.0] ]*16]*16  ]*1 , dtype=tf.float32)
    
    testA = glob('./datasets/{}/*.*'.format(dataset_dir + '/testA'))
    testB = glob('./datasets/{}/*.*'.format(dataset_dir + '/testB'))
    testA.sort()
    testB.sort()
    test_dataPairs = list(zip(testA[:], testB[:]))

    # for epoch in range(args.epoch):
    for epoch in range(epochs):
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
        lr = 0.0002 if epoch < epoch_step else 0.0002*(epochs-epoch)/(epochs-epoch_step)
        discriminator_a.optimizer.lr = lr
        discriminator_b.optimizer.lr = lr
        GAN_a2b.optimizer.lr = lr
        GAN_b2a.optimizer.lr = lr

        for idx in range(0, batch_idxs):
            batch_start_time = time.time()
            #################################################################################################################
            # Load Batch data
            batch_files = list(zip(dataA[idx * batch_size : (idx + 1) * batch_size],
                                   dataB[idx * batch_size : (idx + 1) * batch_size])) ### [('./datasets/horse2zebra/trainA\\n02381460_1766.jpg', './datasets/horse2zebra/trainB\\n02391049_22.jpg')]
            batch_images = [load_train_data(batch_file) for batch_file in batch_files]
            batch_images = np.array(batch_images).astype(np.float32) ### batch_images.shape (1, 256, 256, 6)
            # print(batch_files)

            real_a = batch_images[:,:,:,0:3]
            real_b = batch_images[:,:,:,3:6]
            # print("real_a",real_a[0])
            # cv2.imshow("real_a",real_a[0])
            # cv2.waitKey(0)

            ##################################################################################################################################################################################################################################
            # Update D network
            discriminator_b.trainable = True
            fake_b = generator_a2b(real_a) ### 丟進去要的形式要是 BHWC
            fake_b_concat_real_b = tf.concat([fake_b, real_b] , axis=0)
            discriminator_b.train_on_batch( fake_b_concat_real_b, y_d)

            discriminator_a.trainable = True
            fake_a = generator_b2a(real_b) ### 丟進去要的形式要是 BHWC
            fake_a_concat_real_a = tf.concat([fake_a, real_a] , axis=0)
            discriminator_a.train_on_batch( fake_a_concat_real_a, y_d )
            #################################################################################################################
            # Update G network and record fake outputs
            discriminator_b.trainable = False
            GAN_a2b.train_on_batch( real_a, [real_a, y_g] )

            discriminator_a.trainable = False
            GAN_b2a.train_on_batch( real_b, [real_b, y_g] )
            #################################################################################################################
            ##################################################################################################################################################################################################################################
            # counter += 1 原始寫這邊，我把它調到下面去囉
            cost_time = time.time() - start_time
            hour = cost_time//3600 ; minute = cost_time%3600//60 ; second = cost_time%3600%60
            print(("Epoch: [%2d] [%4d/%4d] b_time: %4.2f, total_time:%2d:%02d:%02d counter:%d" % (
                epoch, idx, batch_idxs, time.time() - batch_start_time, hour, minute, second, counter)))

            if(counter % 100 == 0):
                test_src_path = "datasets/horse2zebra/testA/n02381460_120.jpg"
                test_dst_path = "result/A-n02381460_120_to_B-zibra"
                sample_test_data(test_src_path, test_dst_path,generator_a2b, counter, epoch)


            counter += 1  ### 調到這裡
train()