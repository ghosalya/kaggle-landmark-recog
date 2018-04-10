'''
Implementation of small, fake AlexNet
'''
import tensorflow as tf

def fakealex_model(X, y, set_tensor, numclass=14951):
    '''
    Model Implementation of fake AlexNet in tensorflow
    Ax:
        conv->relu->dropout->conv->relu->dropout
        ->conv->relu->dropout->affine->affine
    '''

    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    set_tensor(Wconv1)
    # resulting in 64x64x32
    Wconv2 = tf.get_variable("Wconv2", shape=[7, 7, 32, 16])
    bconv2 = tf.get_variable("bconv2", shape=[16])
    set_tensor(Wconv2)
    # resulting in 29x29x16
    Wconv3 = tf.get_variable("Wconv3", shape=[3, 3, 16, 64])
    bconv3 = tf.get_variable("bconv3", shape=[64])
    set_tensor(Wconv3)
    # resulting in 9x9x64
    W1 = tf.get_variable("W1", shape=[5184, 7500])
    b1 = tf.get_variable("b1", shape=[7500])
    set_tensor(W1)
    W2 = tf.get_variable("W2", shape=[7500, numclass])
    b2 = tf.get_variable("b2", shape=[numclass])
    set_tensor(W2)

    # define our graph 
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,3,3,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    d1 = tf.nn.dropout(h1, 0.5)

    a2 = tf.nn.conv2d(d1, Wconv2, strides=[1,2,2,1], padding='VALID') + bconv2
    h2 = tf.nn.relu(a2)
    d2 = tf.nn.dropout(h2, 0.5)

    a3 = tf.nn.conv2d(d2, Wconv3, strides=[1,3,3,1], padding='VALID') + bconv3
    h3 = tf.nn.relu(a3)
    d3 = tf.nn.dropout(h3, 0.5)

    d3_flat = tf.reshape(d3,[-1,5184])
    # y_out = tf.matmul(d3_flat,W1) + b1
    aff1 = tf.matmul(d3_flat,W1) + b1
    reff1 = tf.nn.relu(aff1)
    y_out = tf.matmul(reff1, W2) + b2
    return y_out