from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, ReLU, Conv2DTranspose
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
# from tensorflow_addons.layers import InstanceNormalization
tf.keras.backend.set_floatx('float32') ### 這步非常非常重要！用了才可以加速！

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
    def __init__(self,**kwargs):
        super(InstanceNorm_kong, self).__init__(**kwargs)

    def build(self, input_shape):
        depth = input_shape[-1]
        self.scale  = self.add_weight("scale", shape = [depth], initializer=tf.random_normal_initializer(mean=1.0, stddev=0.02), dtype=tf.float32)
        self.offset = self.add_weight("offset", shape = [depth], initializer=tf.constant_initializer(0.0), dtype=tf.float32 )

    def call(self, input):
        mean, variance = tf.nn.moments(input, axes=[1,2], keepdims=True)
        epsilon = tf.constant(1e-5,dtype=tf.float32)
        inv = tf.math.rsqrt(variance + epsilon)
        normalized = (input-mean)*inv
        
        return self.scale*normalized + self.offset
        # return tf.matmul(input, self.kernel)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, c_num, ks=3, s=1):
        super(ResBlock, self).__init__()
        self.ks = ks
        self.conv_1 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c1   = InstanceNorm_kong()
        self.conv_2 = Conv2D( c_num, kernel_size=ks, strides=s, padding="valid")
        self.in_c2   = InstanceNorm_kong()
    
    def call(self, input_tensor):
        p = int( (self.ks-1)/2 )
        x = tf.pad( input_tensor, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)
        x = tf.pad( x, [ [0,0], [p,p], [p,p], [0,0] ], "REFLECT" )
        x = self.conv_2(x)
        x = self.in_c2(x)
        return x + input_tensor

class Discriminator(tf.keras.models.Model):
    def __init__(self,**kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.conv_1 = Conv2D(64  ,   kernel_size=4, strides=2, padding="same")
        self.conv_2 = Conv2D(64*2,   kernel_size=4, strides=2, padding="same")
        self.conv_3 = Conv2D(64*4,   kernel_size=4, strides=2, padding="same")
        self.conv_4 = Conv2D(64*8,   kernel_size=4, strides=2, padding="same")
        self.conv_map = Conv2D(1   ,   kernel_size=4, strides=1, padding="same")

        self.in_c2   = InstanceNorm_kong()
        self.in_c3   = InstanceNorm_kong()
        self.in_c4   = InstanceNorm_kong()
    
    def call(self, input_tensor):
        x = self.conv_1(input_tensor)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_2(x)
        x = self.in_c2(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_3(x)
        x = self.in_c3(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        x = self.conv_4(x)
        x = self.in_c4(x)
        x = tf.nn.leaky_relu(x, alpha=0.2)

        return self.conv_map(x)

class Generator(tf.keras.models.Model):
    def __init__(self,**kwargs):
        super(Generator, self).__init__(**kwargs)
        self.conv1   = Conv2D(64  ,   kernel_size=7, strides=1, padding="valid")
        self.conv2   = Conv2D(64*2,   kernel_size=3, strides=2, padding="same")
        self.conv3   = Conv2D(64*4,   kernel_size=3, strides=2, padding="same")
        self.in_c1   = InstanceNorm_kong()
        self.in_c2   = InstanceNorm_kong()
        self.in_c3   = InstanceNorm_kong()

        self.resb1   = ResBlock(c_num=64*4)
        self.resb2   = ResBlock(c_num=64*4)
        self.resb3   = ResBlock(c_num=64*4)
        self.resb4   = ResBlock(c_num=64*4)
        self.resb5   = ResBlock(c_num=64*4)
        self.resb6   = ResBlock(c_num=64*4)
        self.resb7   = ResBlock(c_num=64*4)
        self.resb8   = ResBlock(c_num=64*4)
        self.resb9   = ResBlock(c_num=64*4)

        self.convT1  = Conv2DTranspose(64*2, kernel_size=3, strides=2, padding="same")
        self.convT2  = Conv2DTranspose(64  , kernel_size=3, strides=2, padding="same")
        self.in_cT1  = InstanceNorm_kong()
        self.in_cT2  = InstanceNorm_kong()
        self.convRGB = Conv2D(3  ,   kernel_size=7, strides=1, padding="valid")

    def call(self, input_tensor):
        x = tf.pad(input_tensor, [[0,0], [3,3], [3,3], [0,0]], "REFLECT")

        ### c1
        x = self.conv1(x)
        x = self.in_c1(x)
        x = tf.nn.relu(x)
        ### c2
        x = self.conv2(x)
        x = self.in_c2(x)
        x = tf.nn.relu(x)
        ### c1
        x = self.conv3(x)
        x = self.in_c3(x)
        x = tf.nn.relu(x)

        x = self.resb1(x)
        x = self.resb2(x)
        x = self.resb3(x)
        x = self.resb4(x)
        x = self.resb5(x)
        x = self.resb6(x)
        x = self.resb7(x)
        x = self.resb8(x)
        x = self.resb9(x)

        x = self.convT1(x)
        x = self.in_cT1(x)
        x = tf.nn.relu(x)

        x = self.convT2(x)
        x = self.in_cT2(x)
        x = tf.nn.relu(x)

        x = tf.pad(x, [[0,0], [3,3], [3,3], [0,0]], "REFLECT")
        x_RGB = self.convRGB(x)
        return tf.nn.tanh(x_RGB)

class CycleGAN(tf.keras.models.Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.discriminator_a = Discriminator(name="D_A")
        self.discriminator_b = Discriminator(name="D_B")
        self.generator_a2b   = Generator(name="G_A2B")
        self.generator_b2a   = Generator(name="G_B2A")

    def call(self, imgA, imgB):
        fake_b       = self.generator_a2b  (imgA)
        fake_b_score = self.discriminator_b(fake_b)
        fake_b_cyc_a = self.generator_b2a  (fake_b)
        real_b_score = self.discriminator_b(imgB)
        
        fake_a       = self.generator_b2a(imgB)
        fake_a_score = self.discriminator_a(fake_a)
        fake_a_cyc_b = self.generator_a2b  (fake_a)
        real_a_score = self.discriminator_a(imgA)

        ### 沒有用 identical loss
        # return fake_b_score, real_b_score, \
        #        fake_a_score, real_a_score, \
        #        fake_b_cyc_a, fake_a_cyc_b

        ### 有用 identical loss
        same_a       = self.generator_b2a(imgA)
        same_b       = self.generator_a2b(imgB)

        return fake_b_score, real_b_score, \
               fake_a_score, real_a_score, \
               fake_b_cyc_a, fake_a_cyc_b, \
               same_a, same_b
class CycleGAN(tf.keras.models.Model):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.discriminator_a = Discriminator(name="D_A")
        self.discriminator_b = Discriminator(name="D_B")
        self.generator_a2b   = Generator(name="G_A2B")
        self.generator_b2a   = Generator(name="G_B2A")

    def call(self, imgA, imgB):
        fake_b       = self.generator_a2b  (imgA)
        fake_b_score = self.discriminator_b(fake_b)
        fake_b_cyc_a = self.generator_b2a  (fake_b)
        real_b_score = self.discriminator_b(imgB)
        
        fake_a       = self.generator_b2a(imgB)
        fake_a_score = self.discriminator_a(fake_a)
        fake_a_cyc_b = self.generator_a2b  (fake_a)
        real_a_score = self.discriminator_a(imgA)

        ### 沒有用 identical loss
        # return fake_b_score, real_b_score, \
        #        fake_a_score, real_a_score, \
        #        fake_b_cyc_a, fake_a_cyc_b

        ### 有用 identical loss
        same_a       = self.generator_b2a(imgA)
        same_b       = self.generator_a2b(imgB)

        return fake_b_score, real_b_score, \
               fake_a_score, real_a_score, \
               fake_b_cyc_a, fake_a_cyc_b, \
               same_a, same_b


@tf.function
def mse_kong(tensor1, tensor2, lamb=tf.constant(1.,tf.float32)):
    loss = tf.reduce_mean( tf.math.square( tensor1 - tensor2 ) )
    return loss * lamb

@tf.function
def mae_kong(tensor1, tensor2, lamb=tf.constant(1.,tf.float32)):
    loss = tf.reduce_mean( tf.math.abs( tensor1 - tensor2 ) )
    return loss * lamb


# def train_step(imgA, imgB, optimizer_G, optimizer_D, cyclegan):
@tf.function
def train_step(imgA, imgB, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B, cyclegan):
    with tf.GradientTape(persistent=True) as tape:
        fake_b_score, real_b_score, \
        fake_a_score, real_a_score, \
        fake_b_cyc_a, fake_a_cyc_b, \
        same_a,  same_b = cyclegan(imgA,imgB)

        loss_rec_a = mae_kong(imgA, fake_b_cyc_a, lamb=tf.constant(10.,tf.float32))
        loss_rec_b = mae_kong(imgB, fake_a_cyc_b, lamb=tf.constant(10.,tf.float32))
        loss_g2d_b = mse_kong(fake_b_score, tf.ones_like(fake_b_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32))
        loss_g2d_a = mse_kong(fake_a_score, tf.ones_like(fake_b_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32))
        loss_same_a = mae_kong(imgA, same_a, lamb=tf.constant(0.5,tf.float32))
        loss_same_b = mae_kong(imgB, same_b, lamb=tf.constant(0.5,tf.float32))
        # g_total_loss = loss_rec_a + loss_rec_b + loss_g2d_b + loss_g2d_a
        G_A2B_loss = loss_rec_a + loss_g2d_b + loss_same_b
        G_B2A_loss = loss_rec_b + loss_g2d_a + loss_same_a

        loss_da_real = mse_kong( real_a_score, tf.ones_like(real_a_score ,dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        loss_da_fake = mse_kong( fake_a_score, tf.zeros_like(fake_a_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        loss_db_real = mse_kong( real_b_score, tf.ones_like(real_b_score ,dtype=tf.float32),   lamb=tf.constant(1.,tf.float32) )
        loss_db_fake = mse_kong( fake_b_score, tf.zeros_like(fake_b_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        # d_total_loss = (loss_da_real+loss_da_fake)/2 + (loss_db_real+loss_db_fake)/2
        D_A_loss = (loss_da_real+loss_da_fake)/2  
        D_B_loss = (loss_db_real+loss_db_fake)/2

    # grad_D = tape.gradient(d_total_loss, cyclegan.discriminator_a.trainable_weights + cyclegan.discriminator_b.trainable_weights)
    # grad_G = tape.gradient(g_total_loss, cyclegan.generator_b2a.  trainable_weights + cyclegan.generator_a2b.  trainable_weights)
    # optimizer_D.apply_gradients( zip(grad_D, cyclegan.discriminator_a.trainable_weights + cyclegan.discriminator_b.trainable_weights)  )
    # optimizer_G.apply_gradients( zip(grad_G, cyclegan.generator_b2a.  trainable_weights + cyclegan.generator_a2b.  trainable_weights)  )
    
    grad_D_A = tape.gradient(D_A_loss, cyclegan.discriminator_a.trainable_weights )
    grad_D_B = tape.gradient(D_B_loss, cyclegan.discriminator_b.trainable_weights )
    grad_G_A2B = tape.gradient(G_A2B_loss, cyclegan.generator_a2b.trainable_weights )
    grad_G_B2A = tape.gradient(G_B2A_loss, cyclegan.generator_b2a.trainable_weights )
    
    
    optimizer_D_A.apply_gradients( zip(grad_D_A, cyclegan.discriminator_a.trainable_weights )  )
    optimizer_D_B.apply_gradients( zip(grad_D_B, cyclegan.discriminator_b.trainable_weights )  )
    optimizer_G_A2B.apply_gradients( zip(grad_G_A2B, cyclegan.generator_a2b.  trainable_weights)  )
    optimizer_G_B2A.apply_gradients( zip(grad_G_B2A, cyclegan.generator_b2a.  trainable_weights)  )


@tf.function
def train_step_tensorboard(imgA, imgB, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B, cyclegan):
    with tf.GradientTape(persistent=True) as tape:
        fake_b_score, real_b_score, \
        fake_a_score, real_a_score, \
        fake_b_cyc_a, fake_a_cyc_b, \
        same_a,  same_b = cyclegan(imgA,imgB)

        # loss_rec_a = mae_kong(imgA, fake_b_cyc_a, lamb=tf.constant(10.,tf.float32))
        # loss_rec_b = mae_kong(imgB, fake_a_cyc_b, lamb=tf.constant(10.,tf.float32))
        # loss_g2d_b = mse_kong(fake_b_score, tf.ones_like(fake_b_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32))
        # loss_g2d_a = mse_kong(fake_a_score, tf.ones_like(fake_b_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32))
        # loss_same_a = mae_kong(imgA, same_a, lamb=tf.constant(0.5,tf.float32))
        # loss_same_b = mae_kong(imgB, same_b, lamb=tf.constant(0.5,tf.float32))
        # G_A2B_loss = loss_rec_a + loss_g2d_b + loss_same_b
        # G_B2A_loss = loss_rec_b + loss_g2d_a + loss_same_a

        # loss_da_real = mse_kong( real_a_score, tf.ones_like(real_a_score ,dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        # loss_da_fake = mse_kong( fake_a_score, tf.zeros_like(fake_a_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        # loss_db_real = mse_kong( real_b_score, tf.ones_like(real_b_score ,dtype=tf.float32),   lamb=tf.constant(1.,tf.float32) )
        # loss_db_fake = mse_kong( fake_b_score, tf.zeros_like(fake_b_score,dtype=tf.float32), lamb=tf.constant(1.,tf.float32) )
        # D_A_loss = (loss_da_real+loss_da_fake)/2  
        # D_B_loss = (loss_db_real+loss_db_fake)/2
    
    # grad_D_A = tape.gradient(D_A_loss, cyclegan.discriminator_a.trainable_weights )
    # grad_D_B = tape.gradient(D_B_loss, cyclegan.discriminator_b.trainable_weights )
    # grad_G_A2B = tape.gradient(G_A2B_loss, cyclegan.generator_a2b.trainable_weights )
    # grad_G_B2A = tape.gradient(G_B2A_loss, cyclegan.generator_b2a.trainable_weights )
    
    
    # optimizer_D_A.apply_gradients( zip(grad_D_A, cyclegan.discriminator_a.trainable_weights )  )
    # optimizer_D_B.apply_gradients( zip(grad_D_B, cyclegan.discriminator_b.trainable_weights )  )
    # optimizer_G_A2B.apply_gradients( zip(grad_G_A2B, cyclegan.generator_a2b.  trainable_weights)  )
    # optimizer_G_B2A.apply_gradients( zip(grad_G_B2A, cyclegan.generator_b2a.  trainable_weights)  )


@tf.function
def create_model(real_a, real_b, cyclegan):
    cyclegan(real_a, real_b)
    return tf.ones(shape = (1,1,1,1))

def log_graph():
    temp = CycleGAN()
    cyclegan = temp.kong_build()
    import time
    real_a = tf.ones(shape=(1,256,256,3), dtype=tf.float32)
    real_b = tf.ones(shape=(1,256,256,3), dtype=tf.float32)

    optimizer_D_A = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_D_B = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_G_A2B = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    optimizer_G_B2A = tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    
    start_time = time.time()
    logdir = "logs"
    writer = tf.summary.create_file_writer(logdir)
    tf.summary.trace_on(graph=True)
    # fake_b_score, real_b_score, \
    #     fake_a_score, real_a_score, \
    #     fake_b_cyc_a, fake_a_cyc_b, \
    #     same_a,  same_b= CycleGAN()(real_a,real_b)

    # with tf.GradientTape(persistent=True) as tape:
    #     fake_b_score, real_b_score, \
    #     fake_a_score, real_a_score, \
    #     fake_b_cyc_a, fake_a_cyc_b, \
    #     same_a, same_b = cyclegan(real_a,real_b)
    # train_step_tensorboard(real_a, real_b, optimizer_G_A2B, optimizer_G_B2A, optimizer_D_A, optimizer_D_B, cyclegan)
    _ = create_model(real_a, real_b, cyclegan)

    with writer.as_default():
        tf.summary.trace_export(name="test_model", step=0)
        writer.flush()
    tf.summary.trace_off()
    print("tensorboard cost time:",time.time()-start_time)
    # cyclegan     .save_weights('cyclegan'    , save_format='tf') 

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

    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)
    g_x = ResBlock(c_num=64*4)(g_x)


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
    discriminator_a.compile(loss="mae", optimizer=optimizer_d_a, )
    discriminator_b.compile(loss="mae", optimizer=optimizer_d_b)
    GAN_b2a.compile(loss=["mae","mae"], optimizer=optimizer_GAN_b2a, loss_weights=[10,1])
    GAN_a2b.compile(loss=["mae","mae"], optimizer=optimizer_GAN_a2b, loss_weights=[10,1])


    # optimizer_D = Adam(lr=0.0002, beta_1=0.5)
    # optimizer_G = Adam(lr=0.0002, beta_1=0.5)
    # discriminator_a.compile(loss="mae", optimizer=optimizer_D, )
    # discriminator_b.compile(loss="mae", optimizer=optimizer_D)
    # GAN_b2a.compile(loss=["mae","mae"], optimizer=optimizer_G, loss_weights=[10,1])
    # GAN_a2b.compile(loss=["mae","mae"], optimizer=optimizer_G, loss_weights=[10,1])

    discriminator_a.trainable = False
    discriminator_b.trainable = False
    ### debug用
    # generator_a2b.trainable = False
    # generator_b2a.trainable = False
    # GAN_b2a.trainable = False
    # GAN_a2b.trainable = False
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

    # discriminator_a, discriminator_b, generator_a2b, generator_b2a, GAN_b2a, GAN_a2b = build_CycleGAN()
    # discriminator_a.save('discriminator_a.h5') 
    # generator_a2b.save('generator_a2b.h5') 

    # cyclegan = CycleGAN()
    # log_graph()

    print("finish")