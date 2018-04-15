'''
Implementation of small, fake AlexNet
'''
import math
import tensorflow as tf

def fakealex_model(X, y, set_tensor, numclass=14951, size=128):
    '''
    Model Implementation of fake AlexNet in tensorflow
    Ax:
        conv->relu->dropout->conv->relu->dropout
        ->conv->relu->dropout->affine->affine
    '''

    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[11, 11, 3, 96])
    bconv1 = tf.get_variable("bconv1", shape=[96])
    set_tensor('Wconv1',Wconv1) # maxpool
    Wconv2 = tf.get_variable("Wconv2", shape=[5, 5, 96, 256])
    bconv2 = tf.get_variable("bconv2", shape=[256])
    set_tensor('Wconv2',Wconv2) # maxpool
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 256, 384])
    bconv3 = tf.get_variable("bconv3", shape=[384])
    set_tensor('Wconv3',Wconv3)
    Wconv4 = tf.get_variable("Wconv4", shape=[3, 3, 384, 384])
    bconv4 = tf.get_variable("bconv4", shape=[384])
    set_tensor('Wconv4',Wconv4)
    Wconv5 = tf.get_variable("Wconv5", shape=[3, 3, 384, 256])
    bconv5 = tf.get_variable("bconv5", shape=[256])
    set_tensor('Wconv5',Wconv5) #maxpool

    flat_size = int(1.5 * (size // (4*2*2))**2 * 256) # TODO: DONT PUT 1.5*
    # flat_dim = math.ceil(size / 8)
    # flat_size = flat_dim**2 * 256

    W1 = tf.get_variable("W1", shape=[flat_size, 2048])
    b1 = tf.get_variable("b1", shape=[2048])
    set_tensor('W1',W1)
    W2 = tf.get_variable("W2", shape=[2048, numclass])
    b2 = tf.get_variable("b2", shape=[numclass])
    set_tensor('W2',W2)

    # define our graph 
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,4,4,1], padding='SAME') + bconv1
    ma1 = tf.nn.max_pool(a1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    h1 = tf.nn.relu(ma1)
    d1 = tf.nn.dropout(h1, 0.5)

    a2 = tf.nn.conv2d(d1, Wconv2, strides=[1,1,1,1], padding='SAME') + bconv2
    ma2 = tf.nn.max_pool(a2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    h2 = tf.nn.relu(ma2)
    d2 = tf.nn.dropout(h2, 0.5)

    a3 = tf.nn.conv2d(d2, Wconv3, strides=[1,1,1,1], padding='SAME') + bconv3
    h3 = tf.nn.relu(a3)
    d3 = tf.nn.dropout(h3, 0.5)

    a4 = tf.nn.conv2d(d3, Wconv4, strides=[1,1,1,1], padding='SAME') + bconv4
    h4 = tf.nn.relu(a4)
    d4 = tf.nn.dropout(h4, 0.5)

    a5 = tf.nn.conv2d(d4, Wconv5, strides=[1,1,1,1], padding='SAME') + bconv5
    ma5 = tf.nn.max_pool(a5, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    h5 = tf.nn.relu(ma5)
    d5 = tf.nn.dropout(h5, 0.5)


    d5_flat = tf.reshape(d3,[-1,flat_size])
    # y_out = tf.matmul(d3_flat,W1) + b1
    aff1 = tf.matmul(d5_flat,W1) + b1
    reff1 = tf.nn.relu(aff1)
    y_out = tf.matmul(reff1, W2) + b2
    return y_out