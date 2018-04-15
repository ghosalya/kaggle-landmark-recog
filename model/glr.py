'''
Based on the claim that a landmark has
both 'global' features (figures, outline etc)
and 'local' features (texture, windows etc)

global -> shallow, big reception layer, followed by dimensionality reduction
local -> small, deep reception layer

concatenate to affine
hopefully not too big X(
'''

import tensorflow as tf

def glr_local(X, y, set_tensor, numclass=14951):
    # small, 3 layer convnet
    Wconv1 = tf.get_variable("local-Wconv1", shape=[3, 3, 3, 16])
    bconv1 = tf.get_variable("local-bconv1", shape=[16])
    set_tensor('local-Wconv1',Wconv1)
    Wconv2 = tf.get_variable("local-Wconv2", shape=[4, 4, 16, 32])
    bconv2 = tf.get_variable("local-bconv2", shape=[32])
    set_tensor('local-Wconv2',Wconv2)
    Wconv3 = tf.get_variable("local-Wconv3", shape=[5, 5, 32, 64])
    bconv3 = tf.get_variable("local-bconv3", shape=[64])
    set_tensor('local-Wconv3',Wconv3)

    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,3,3,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    d1 = tf.nn.dropout(h1, 0.5)
    a2 = tf.nn.conv2d(d1, Wconv2, strides=[1,2,2,1], padding='VALID') + bconv2
    h2 = tf.nn.relu(a2)
    d2 = tf.nn.dropout(h2, 0.5)
    a3 = tf.nn.conv2d(d2, Wconv3, strides=[1,3,3,1], padding='VALID') + bconv3
    h3 = tf.nn.relu(a3)
    d3 = tf.nn.dropout(h3, 0.5)
    # output should be 6x6x64
    return d3

def glr_global(X, y, set_tensor, numclass=14951):
    # one layer convnet with dim reduction
    Wconv1 = tf.get_variable("global-Wconv1", shape=[32, 32, 3, 96])
    bconv1 = tf.get_variable("global-bconv1", shape=[96])
    set_tensor('local-Wconv1',Wconv1)

    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,8,8,1], padding='VALID') + bconv1
    # h1 = tf.nn.relu(a1)
    h1 = tf.nn.max_pool(a1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    d1 = tf.nn.dropout(h1, 0.5)
    # result should be 6x6x96
    return d1

def glr_model(X, y, set_tensor, numclass=14951, saver=None):
    local = glr_local(X, y, set_tensor)
    globl = glr_global(X, y, set_tensor)

    concoct = tf.concat([local, globl], -1)
    # concat everything and affine twice
    hidden_dim = 250
    W1 = tf.get_variable("W1", shape=[5760, hidden_dim])
    b1 = tf.get_variable("b1", shape=[hidden_dim])
    set_tensor('W1', W1)
    W2 = tf.get_variable("W2", shape=[hidden_dim, numclass])
    b2 = tf.get_variable("b2", shape=[numclass])
    set_tensor('W2', W2)

    concoct_flat = tf.reshape(concoct, [-1,5760])
    y1 = tf.matmul(concoct_flat, W1) + b1
    y1r = tf.nn.relu(y1)
    y2 = tf.matmul(y1r, W2) + b2
    return y2