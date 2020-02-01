from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
# from tensorflow_addons.layers import InstanceNormalization

def instance_norm(in_x, name="instance_norm"):    
    depth = in_x.get_shape()[3]
    scale = tf.Variable(tf.random.normal(shape=[depth],mean=1.0, stddev=0.02), dtype=tf.float32)
    #print(scale)
    offset = tf.Variable(tf.zeros(shape=[depth]))
    mean, variance = tf.nn.moments(in_x, axes=[1,2], keepdims=True)
    # print("mean",mean)
    # print("variance",variance)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (in_x-mean)*inv
    return scale*normalized + offset


class InstanceNorm_kong(tf.keras.layers.Layer):
    def __init__(self):
        super(InstanceNorm_kong, self).__init__()

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale  = self.add_variable("scale", shape = [depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02))
        self.offset = self.add_variable("ofset", shape = [depth], initializer=tf.constant_initializer(0.0) )

    def call(self, input):
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)
        epsilon = 1e-5
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        
        return self.scale*normalized + self.offset
        # return tf.matmul(input, self.kernel)


def build_D(d_in, name = ""):
    
    d_x  = Conv2D(64  , kernel_size=4, strides=2, padding="same")(d_in)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_x  = Conv2D(64*2, kernel_size=4, strides=2, padding="same")(d_x)
    d_x  = InstanceNorm_kong()(d_x)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_x  = Conv2D(64*4, kernel_size=4, strides=2, padding="same")(d_x)
    d_x  = InstanceNorm_kong()(d_x)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_x  = Conv2D(64*8, kernel_size=4, strides=2, padding="same")(d_x)
    d_x  = InstanceNorm_kong()(d_x)
    d_x  = LeakyReLU(alpha = 0.2)(d_x)

    d_score_map  = Conv2D(1, kernel_size=4, strides=1, padding="same")(d_x)

    discriminator = Model(d_in,d_score_map, name=name)
    return discriminator
################################################################################


def build_G(g_in,name = ""):
    def residule_block(x, c_num, ks=3, s=1):
        p = int( (ks-1)/2 )
        y = tf.pad( x, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        y = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")(y)
        y = InstanceNorm_kong()(y)
        y = ReLU()(y)
        y = tf.pad( y, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT")
        y = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")(y)
        y = InstanceNorm_kong()(y)
        return y + x

    
    g_x = tf.pad(g_in, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")

    ### c1
    g_x = Conv2D(64  , kernel_size=7, strides=1, padding="valid")(g_x)
    g_x = InstanceNorm_kong()(g_x)
    g_x = ReLU()(g_x)
    ### c2
    g_x = Conv2D(64*2, kernel_size=3, strides=2, padding="same")(g_x)
    g_x = InstanceNorm_kong()(g_x)
    g_x = ReLU()(g_x)
    ### c3
    g_x = Conv2D(64*4, kernel_size=3, strides=2, padding="same")(g_x)
    g_x = InstanceNorm_kong()(g_x)
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
    g_x = InstanceNorm_kong()(g_x)
    g_x = ReLU()(g_x)

    g_x = Conv2DTranspose(64  , kernel_size=3, strides=2, padding="same")(g_x)
    g_x = InstanceNorm_kong()(g_x)
    g_x = ReLU()(g_x)

    g_x = tf.pad(g_x, [ [0,0], [3,3], [3,3], [0,0] ], "REFLECT")
    g_img = Conv2D(3, kernel_size=7, strides=1, padding="valid",activation="tanh")(g_x)
    generator = Model(g_in,g_img,name=name)
    return generator
# print(g_img)


def build_CycleGAN():
    d_in_a = Input(shape=(None,None,3), name="D_A_IN")
    d_in_b = Input(shape=(None,None,3), name="D_B_IN")
    discriminator_a = build_D(d_in_a, name="D_A")
    discriminator_b = build_D(d_in_b, name="D_B")


    g_in_a = Input(shape=(None, None, 3), name="G_A_IN")
    g_in_b = Input(shape=(None, None, 3), name="G_B_IN")
    generator_a2b = build_G(g_in_a,name = "G_A2B")
    generator_b2a = build_G(g_in_b,name = "G_B2A")

    fake_b       = generator_a2b(g_in_a)
    fake_b_score = discriminator_b(fake_b)
    fake_b_cyc_a = generator_b2a(fake_b)
    GAN_a2b = Model(inputs=g_in_a, outputs=[fake_b_cyc_a, fake_b_score],name="GAN_A2B")

    fake_a = generator_b2a(g_in_b)
    fake_a_score = discriminator_a(fake_a)
    fake_a_cyc_b = generator_a2b(fake_a)
    GAN_b2a = Model(inputs=g_in_b, outputs=[fake_a_cyc_b, fake_a_score],name="GAN_B2A")

    optimizer_d_a = Adam(lr=0.0002, beta_1=0.5)
    optimizer_d_b = Adam(lr=0.0002, beta_1=0.5)
    optimizer_GAN_b2a = Adam(lr=0.0002, beta_1=0.5)
    optimizer_GAN_a2b = Adam(lr=0.0002, beta_1=0.5)
    
    ### debug用
    # generator_a2b.trainable = False
    # generator_b2a.trainable = False
    # GAN_b2a.trainable = False
    # GAN_a2b.trainable = False

    discriminator_a.compile(loss="mae", optimizer=optimizer_d_a, )
    discriminator_b.compile(loss="mae", optimizer=optimizer_d_b)
    GAN_b2a.compile(loss=["mae","mae"], optimizer=optimizer_GAN_b2a, loss_weights=[10,1])
    GAN_a2b.compile(loss=["mae","mae"], optimizer=optimizer_GAN_a2b, loss_weights=[10,1])

    discriminator_a.trainable = False
    discriminator_b.trainable = False
    
    return discriminator_a, discriminator_b, generator_a2b, generator_b2a, GAN_b2a, GAN_a2b

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

    discriminator_a, discriminator_b, generator_a2b, generator_b2a, GAN_b2a, GAN_a2b = build_CycleGAN()
    discriminator_a.save('discriminator_a.h5') 
# d_x  = InstanceNormalization(axis=3, 
#                                 center=True, 
#                                 scale=True,
#                                 beta_initializer="random_uniform",
#                                 gamma_initializer="random_uniform")(d_x)
    print("finish")