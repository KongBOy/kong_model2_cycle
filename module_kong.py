from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
# from tensorflow_addons.layers import InstanceNormalization
def build_D(d_in):
    
    d_x  = Conv2D(64  , kernel_size=4, strides=2, padding="same")(d_in)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_x  = Conv2D(64*2, kernel_size=4, strides=2, padding="same")(d_x)
    d_x  = BatchNormalization()(d_x)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_x  = Conv2D(64*4, kernel_size=4, strides=2, padding="same")(d_x)
    d_x  = BatchNormalization()(d_x)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_x  = Conv2D(64*8, kernel_size=4, strides=2, padding="same")(d_x)
    d_x  = BatchNormalization()(d_x)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_score_map  = Conv2D(1, kernel_size=4, strides=1, padding="same")(d_x)

    discriminator = Model(d_in,d_score_map)
    return discriminator
################################################################################


def build_G(g_in):
    def residule_block(x, c_num, ks=3, s=1):
        p = int( (ks-1)/2 )
        y = tf.pad( x, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        y = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")(y)
        y = BatchNormalization()(y)
        y = ReLU()(y)
        y = tf.pad( y, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT")
        y = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")(y)
        y = BatchNormalization()(y)
        return y + x

    
    g_x = tf.pad(g_in, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

    ### c1
    g_x = Conv2D(64  , kernel_size=7, strides=1, padding="valid")(g_x)
    g_x = BatchNormalization( )(g_x)
    g_x = ReLU()(g_x)
    ### c2
    g_x = Conv2D(64*2, kernel_size=3, strides=2, padding="same")(g_x)
    g_x = BatchNormalization( )(g_x)
    g_x = ReLU()(g_x)
    ### c3
    g_x = Conv2D(64*4, kernel_size=3, strides=2, padding="same")(g_x)
    g_x = BatchNormalization( )(g_x)
    g_x = ReLU()(g_x)

    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)
    g_x = residule_block(g_x, c_num=64*4)

    g_x = Conv2DTranspose(64*2, kernel_size=3, strides=2, padding="same")(g_x)
    g_x = BatchNormalization()(g_x)
    g_x = ReLU()(g_x)

    g_x = Conv2DTranspose(64  , kernel_size=3, strides=2, padding="same")(g_x)
    g_x = BatchNormalization()(g_x)
    g_x = ReLU()(g_x)

    g_x = tf.pad(g_x, [ [0,0], [3,3], [3,3], [0,0] ], "REFLECT")
    g_img = Conv2D(3, kernel_size=7, strides=1, padding="valid",activation="tanh")(g_x)
    generator = Model(g_in,g_img)
    return generator
# print(g_img)


def build_CycleGAN():
    d_in_a = Input(shape=(None,None,3))
    d_in_b = Input(shape=(None,None,3))
    discriminator_a = build_D(d_in_a)
    discriminator_b = build_D(d_in_b)
    discriminator_a.trainable = False
    discriminator_b.trainable = False

    g_in_a = Input(shape=(None, None, 3))
    g_in_b = Input(shape=(None, None, 3))
    generator_a2b = build_G(g_in_a)
    generator_b2a = build_G(g_in_b)

    gen_b = generator_a2b(g_in_a)
    fake_b_score = discriminator_b(gen_b)
    GAN_b = Model(inputs=g_in_a, outputs=[gen_b, fake_b_score])


    gen_a = generator_b2a(g_in_b)
    fake_a_score = discriminator_a(gen_a)
    GAN_a = Model(inputs=g_in_b, outputs=[gen_a, fake_a_score])

    optimizer_d_a = Adam(lr=0.0002, beta_1=0.5)
    optimizer_d_b = Adam(lr=0.0002, beta_1=0.5)
    optimizer_GAN_a = Adam(lr=0.0002, beta_1=0.5)
    optimizer_GAN_b = Adam(lr=0.0002, beta_1=0.5)
    
    def d_loss_function(y_true, y_pred):
        return tf.math.reduce_mean(tf.abs(y_true - y_pred))/2


    discriminator_a.compile(loss="mae", optimizer=optimizer_d_a)
    discriminator_b.compile(loss="mae", optimizer=optimizer_d_b)
    GAN_a.compile(loss=["mae","mae"], optimizer=optimizer_GAN_a)
    GAN_b.compile(loss=["mae","mae"], optimizer=optimizer_GAN_b)
    
    return discriminator_a, discriminator_b, generator_a2b, generator_b2a, GAN_a, GAN_b

if(__name__ == "__main__"):
    import numpy as np
    # generator = build_G()
    # img_g = np.ones( shape=(1,16,16,3), dtype=np.float32)
    # out_g = generator(img_g)
    # print("out_g.numpy()",out_g.numpy())

    # discriminator = build_D()
    # img_d = np.ones(shape=(1,16,16,6),dtype=np.float32)
    # out_d = discriminator(img_d)
    # print("out_d.numpy()",out_d.numpy())

    build_CycleGAN()
# d_x  = InstanceNormalization(axis=3, 
#                                 center=True, 
#                                 scale=True,
#                                 beta_initializer="random_uniform",
#                                 gamma_initializer="random_uniform")(d_x)
    print("finish")