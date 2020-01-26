import tensorflow as tf 
from tensorflow.python.keras.layers import Input, Conv2D
from tensorflow.python.keras.models import Model
def instance_norm(in_x, name="instance_norm"):

    depth = in_x.get_shape()[3]
    scale = tf.Variable(tf.random.normal(shape=[depth],mean=1.0, stddev=0.02), dtype=tf.float32)
    print(scale)
    offset = tf.Variable(tf.zeros(shape=[depth]))
    mean, variance = tf.nn.moments(in_x, axes=[1,2], keepdims=True)
    print("mean",mean)
    print("variance",variance)
    epsilon = 1e-5
    inv = tf.math.rsqrt(variance + epsilon)
    normalized = (in_x-mean)*inv
    return scale*normalized + offset
    # scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
    # offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
    # mean, variance = tf.nn.moments(input, axes=[1,2], keep_dims=True)
    # epsilon = 1e-5
    # inv = tf.rsqrt(variance + epsilon)
    # normalized = (input-mean)*inv
    # return scale*normalized + offset


g_in = Input(shape=(None, None, 1))
g_x = Conv2D(2  , kernel_size=3, strides=1, padding="valid")(g_in)
g_x = instance_norm(g_x)
generator = Model(g_in, g_x)


import numpy as np 
img = np.arange( 32, dtype=np.float32).reshape(2,4,4,1)
print(generator(img).numpy())
print(generator(img[0:1,:,:,:]).numpy())
print("finish")