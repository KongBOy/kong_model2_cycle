from glob import glob
import numpy as np 
# from imageio import imread
from scipy.misc import imread
import scipy.misc
import time

from module_kong import build_CycleGAN, CycleGAN, train_step

import tensorflow as tf

from build_dataset_combine import *
import shutil


def augmentation(img, load_size=286, fine_size=256):
    img = scipy.misc.imresize(img, [load_size, load_size])
    h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
    img = img[h1:h1+fine_size, w1:w1+fine_size]
    
    if np.random.random() > 0.5:
        img = np.fliplr(img)
    return img

def combine_dataset(datasetA, datasetB):
    min_amount = np.minimum(len(datasetA), len(datasetB))
    combine_list = []
    for i in range(min_amount):
        combine_list.append( [ datasetA[i], datasetB[i]  ] )
    return np.array(combine_list)



def load_all_data(file_names):
    # file_names = [ file_name for file_name in os.listdir(img_dir) if ".jpg" in file_name.lower() ]
    imgs = [ augmentation(imread(file_name, mode="RGB")) for file_name in file_names ]
    imgs = (np.array(imgs) / 255)*2 -1
    return imgs

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
    # discriminator_a, discriminator_b, generator_a2b, generator_b2a, GAN_b2a, GAN_a2b = build_CycleGAN()
    cyclegan = CycleGAN()
    # optimizer_D = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    # optimizer_G = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_D_A = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_D_B = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_G_A2B = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_G_B2A = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    print("create model cost time:",time.time() - start_time)
    ##############################################################################################################

    Check_dir_exist_and_build_new_dir("result/")
    Check_dir_exist_and_build_new_dir("result/img")
    shutil.copy("load_and_train.py","result/load_and_train.py")
    shutil.copy("module_kong.py",   "result/module_kong.py")

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
    # test_dataPairs = list(zip(testA[:], testB[:]))

    for epoch in range(epochs):
        train_size  = 1e8
        batch_size  = 1

        dataA = glob('./datasets/{}/*.*'.format(dataset_dir + '/trainA'))
        dataB = glob('./datasets/{}/*.*'.format(dataset_dir + '/trainB'))
        np.random.shuffle(dataA)
        np.random.shuffle(dataB)

        dataA_imgs = load_all_data(dataA)
        dataB_imgs = load_all_data(dataB)
        # combine_imgs = combine_dataset(dataA_imgs, dataB_imgs)
        
        datasetA = tf.data.Dataset.from_tensor_slices(dataA_imgs).prefetch(1)
        datasetB = tf.data.Dataset.from_tensor_slices(dataB_imgs).prefetch(1)
        imgA = iter(datasetA)
        imgB = iter(datasetB)
        # datasetC = tf.data.Dataset.from_tensor_slices(combine_imgs).prefetch(5)
        # pairs = iter(datasetC)

        
        batch_idxs = min(min(len(dataA), len(dataB)), train_size) // batch_size
        # batch_idxs = len(combine_imgs) // batch_size

        # lr = args.lr if epoch < args.epoch_step else args.lr*(args.epoch-epoch)/(args.epoch-args.epoch_step)
        lr = 0.0002 if epoch < epoch_step else 0.0002*(epochs-epoch)/(epochs-epoch_step)
        # optimizer_G.lr = lr
        # optimizer_D.lr = lr
        optimizer_D_A.lr = lr
        optimizer_D_B.lr = lr
        optimizer_G_A2B.lr = lr
        optimizer_G_B2A.lr = lr
        
        # discriminator_a.optimizer.lr = lr
        # discriminator_b.optimizer.lr = lr
        # GAN_a2b.optimizer.lr = lr
        # GAN_b2a.optimizer.lr = lr
        ##############################################################################################################

        import cv2
        for idx in range(0, batch_idxs):
            batch_start_time = time.time()
            #################################################################################################################
            # Load Batch data   
            real_a = tf.reshape( tf.cast(next(imgA), tf.float32), shape=(1,256,256,3))    
            real_b = tf.reshape( tf.cast(next(imgB), tf.float32), shape=(1,256,256,3))
            
            # print( ((real_a.numpy()+1)*125).astype(np.uint8) )
            # cv2.imshow("real_a", ((real_a.numpy()+1)*125).astype(np.uint8).reshape(256,256,3))
            # cv2.imshow("real_b", ((real_b.numpy()+1)*125).astype(np.uint8).reshape(256,256,3))
            # cv2.waitKey(0)

            # train_step(real_a, real_b, optimizer_G, optimizer_D, cyclegan)
            train_step(real_a, real_b, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B, cyclegan)
            # pair_img = tf.cast(next(pairs), tf.float32)
            # real_a = pair_img[0:1, ...]
            # real_b = pair_img[1:2, ...]
            
            # import cv2
            # print("real_a",real_a[0])
            # cv2.imshow("real_a",real_a[0])
            # cv2.waitKey(0)

            ##################################################################################################################################################################################################################################
            # Update D network
            # discriminator_b.trainable = True
            # fake_b = generator_a2b(real_a) ### 丟進去要的形式要是 BHWC
            # fake_b_concat_real_b = tf.concat([fake_b, real_b] , axis=0)
            # discriminator_b.train_on_batch( fake_b_concat_real_b, y_d)

            # discriminator_a.trainable = True
            # fake_a = generator_b2a(real_b) ### 丟進去要的形式要是 BHWC
            # fake_a_concat_real_a = tf.concat([fake_a, real_a] , axis=0)
            # discriminator_a.train_on_batch( fake_a_concat_real_a, y_d )
            # # #################################################################################################################
            # # Update G network and record fake outputs
            # discriminator_b.trainable = False
            # GAN_a2b.train_on_batch( real_a, [real_a, y_g] )

            # discriminator_a.trainable = False
            # GAN_b2a.train_on_batch( real_b, [real_b, y_g] )
            # #################################################################################################################
            ##################################################################################################################################################################################################################################
            cost_time = time.time() - start_time
            hour = cost_time//3600; minute = cost_time%3600//60; second = cost_time%3600%60
            print(("Epoch: [%2d] [%4d/%4d] b_time: %4.2f, total_time:%2d:%02d:%02d counter:%d" % (
                epoch, idx, batch_idxs, time.time() - batch_start_time, hour, minute, second, counter)))

            if(counter % 100 == 0):
                test_src_path = "datasets/horse2zebra/testA/n02381460_120.jpg"
                test_dst_path = "result/A-n02381460_120_to_B-zibra"
                sample_test_data(test_src_path, test_dst_path, cyclegan.generator_a2b, counter, epoch)


            counter += 1  ### 調到這裡
train()