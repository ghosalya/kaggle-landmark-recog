'''
Resnet implementation
'''
import random, math
import tensorflow as tf

def resnet_mod(y_in, set_tensor, dim=128, is_training=True, name=None, kernel=64):
    if name is None:
        name = ''.join(random.choice('AEIUORSTGHZYX') for _ in range(3))

    Wconv1 = tf.get_variable("{}-Wconv1".format(name), shape=[3, 3, kernel, kernel])
    bconv1 = tf.get_variable("{}-bconv1".format(name), shape=[kernel])
    scale1 = tf.Variable(tf.ones([64]))
    beta1 = tf.Variable(tf.zeros([64]))
    set_tensor('{}-Wconv1'.format(name), Wconv1)

    Wconv2 = tf.get_variable("{}-Wconv2".format(name), shape=[3, 3, kernel, kernel])
    bconv2 = tf.get_variable("{}-bconv2".format(name), shape=[kernel])
    scale2 = tf.Variable(tf.ones([dim]))
    beta2 = tf.Variable(tf.zeros([dim]))
    set_tensor('{}-Wconv2'.format(name),Wconv2)

    c1 = tf.nn.conv2d(y_in, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    h1 = tf.contrib.layers.batch_norm(c1, 
                                          center=True, scale=True, 
                                          is_training=is_training,
                                          scope='bn1_{}'.format(name))
    r1 = tf.nn.relu(h1)
    c2 = tf.nn.conv2d(r1, Wconv2, strides=[1,1,1,1], padding='SAME') + bconv2
    h2 = tf.contrib.layers.batch_norm(c2, 
                                          center=True, scale=True, 
                                          is_training=is_training,
                                          scope='bn2_{}'.format(name))
    y_out = y_in + h2
    return y_out

def resnet_model(X, y, set_tensor, size=128, numclass=14951, is_training=True):
    '''
    Model Implementation of Resnet in tensorflow
    Ax:
        conv->relu->dropout->conv->relu->dropout
        ->conv->relu->dropout->affine->affine
    '''
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 64])
    bconv1 = tf.get_variable("bconv1", shape=[64])
    set_tensor('Wconv1', Wconv1)
    Wconv2 = tf.get_variable("Wconv2", shape=[3, 3, 64, 128])
    bconv2 = tf.get_variable("bconv2", shape=[128])
    set_tensor('Wconv2', Wconv2)
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 128, 256])
    bconv3 = tf.get_variable("bconv3", shape=[256])
    set_tensor('Wconv3', Wconv3)
    Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 256, 512])
    bconv4 = tf.get_variable("bconv4", shape=[512])
    set_tensor('Wconv4', Wconv4)

    # for now this code assumes size 128
    # dim = 128
    flat_dim = size

    initensor = tf.nn.conv2d(X, Wconv1, strides=[1,2,2,1], padding='SAME') + bconv1
    init_pool = tf.nn.max_pool(initensor, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')

    flat_dim = size // 4

    res1 = resnet_mod(init_pool, set_tensor, is_training=is_training, name='resnet1', kernel=64)
    rd1 = tf.nn.dropout(res1, 0.5)
    res2 = resnet_mod(rd1, set_tensor, is_training=is_training,  name='resnet2', kernel=64)
    rd2 = tf.nn.dropout(res2, 0.5)
    res3 = resnet_mod(rd2, set_tensor, is_training=is_training,  name='resnet3', kernel=64)

    outres1 = tf.nn.conv2d(res3, Wconv2, strides=[1,2,2,1], padding='SAME') + bconv2
    # outres1_pool = tf.nn.max_pool(outres1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    flat_dim = size // 2

    res4 = resnet_mod(outres1, set_tensor, is_training=is_training,  name='resnet4', kernel=128)
    rd4 = tf.nn.dropout(res4, 0.5)
    res5 = resnet_mod(rd4, set_tensor, is_training=is_training,  name='resnet5', kernel=128)
    rd5 = tf.nn.dropout(res5, 0.5)
    res6 = resnet_mod(rd5, set_tensor, is_training=is_training,  name='resnet6', kernel=128)
    # rd6 = tf.nn.dropout(res6, 0.5)

    outres2 = tf.nn.conv2d(res6, Wconv3, strides=[1,2,2,1], padding='SAME') + bconv3
    flat_dim = size // 2

    res7 = resnet_mod(outres2, set_tensor, is_training=is_training,  name='resnet7', kernel=256)
    rd7 = tf.nn.dropout(res7, 0.5)
    res8 = resnet_mod(rd7, set_tensor, is_training=is_training,  name='resnet8', kernel=256)
    rd8 = tf.nn.dropout(res8, 0.5)
    res9 = resnet_mod(rd8, set_tensor, is_training=is_training,  name='resnet9', kernel=256)
    rd9 = tf.nn.dropout(res9, 0.5)
    res10 = resnet_mod(rd9, set_tensor, is_training=is_training,  name='resnet10', kernel=256)
    rd10 = tf.nn.dropout(res10, 0.5)
    res11 = resnet_mod(rd10, set_tensor, is_training=is_training,  name='resnet11', kernel=256)
    rd11 = tf.nn.dropout(res11, 0.5)

    outres3 = tf.nn.conv2d(rd11, Wconv4, strides=[1,2,2,1], padding='SAME') + bconv4
    flat_dim = size // 2

    res12 = resnet_mod(outres3, set_tensor, is_training=is_training,  name='resnet12', kernel=512)
    rd12 = tf.nn.dropout(res12, 0.5)
    res13 = resnet_mod(rd12, set_tensor, is_training=is_training,  name='resnet13', kernel=512)
    rd13 = tf.nn.dropout(res13, 0.5)

    all_pooled = tf.nn.pool(rd13, [3,3], 'AVG', padding="VALID")
    flat_dim = size // 3

    # flat_dim = math.ceil((((size - 6) / 3) - 6) / 2)
    flat_dim = size 
    # flat_size = flat_dim**2 * 16 # 5148
    flat_size = 512 * size // 8

    W1 = tf.get_variable("W1", shape=[flat_size, numclass])
    b1 = tf.get_variable("b1", shape=[numclass])

    res_flat = tf.reshape(rd13,[-1,flat_size])
    y_out = tf.matmul(res_flat,W1) + b1
    return y_out

