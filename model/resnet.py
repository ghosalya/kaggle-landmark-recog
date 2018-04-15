'''
Resnet implementation
'''
import random
import tensorflow as tf
from model.simple import simple_model

def resnet_mod(y_in, dim, set_tensor, name=None, saver=None):
    if name is None:
        name = ''.join(random.choice('AEIUORSTGHZYX') for _ in range(3))

    Wconv1 = tf.get_variable("{}-Wconv1".format(name), shape=[1, 1, 3, 28])
    bconv1 = tf.get_variable("{}-bconv1".format(name), shape=[28])
    scale1 = tf.Variable(tf.ones([64]))
    beta1 = tf.Variable(tf.zeros([64]))
    set_tensor('{}-Wconv1'.format(name), Wconv1)
    if saver is not None:
        saver.var_list.append(Wconv1)
        saver.var_list.append(bconv1)

    Wconv2 = tf.get_variable("{}-Wconv2".format(name), shape=[3, 3, 28, 28])
    bconv2 = tf.get_variable("{}-bconv2".format(name), shape=[28])
    scale2 = tf.Variable(tf.ones([dim]))
    beta2 = tf.Variable(tf.zeros([dim]))
    set_tensor('{}-Wconv2'.format(name),Wconv2)
    if saver is not None:
        saver.var_list.append(Wconv2)
        saver.var_list.append(bconv2)

    Wconv3 = tf.get_variable("{}-Wconv3".format(name), shape=[1, 1, 28, 3])
    bconv3 = tf.get_variable("{}-bconv3".format(name), shape=[3])
    scale3 = tf.Variable(tf.ones([dim]))
    beta3 = tf.Variable(tf.zeros([dim]))
    set_tensor('{}-Wconv3'.format(name),Wconv3)
    if saver is not None:
        saver.var_list.append(Wconv3)
        saver.var_list.append(bconv3)

    c1 = tf.nn.conv2d(y_in, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    # bn1mean, bn1var = tf.nn.moments(c1, [0])
    # bn1 = tf.nn.batch_normalization(c1, bn1mean, bn1var, beta1, scale1, 1e-3)
    r1 = tf.nn.relu(c1)
    c2 = tf.nn.conv2d(r1, Wconv2, strides=[1,1,1,1], padding='SAME') + bconv2
    # bn2mean, bn2var = tf.nn.moments(c2, [0])
    # bn2 = tf.nn.batch_normalization(c2, bn2mean, bn2var, beta2, scale2, 1e-3)
    r2 = tf.nn.relu(c2)
    c3 = tf.nn.conv2d(r2, Wconv3, strides=[1,1,1,1], padding='SAME') + bconv3
    y_out = y_in + c3

    return y_out

def resnet_model(X, y, set_tensor, numclass=14951, saver=None):
    '''
    Model Implementation of Resnet in tensorflow
    Ax:
        conv->relu->dropout->conv->relu->dropout
        ->conv->relu->dropout->affine->affine
    '''

    # for now this code assumes size 128
    dim = 128
    res1 = resnet_mod(X, dim, set_tensor, name='resnet1')
    rd1 = tf.nn.dropout(res1, 0.5)
    res2 = resnet_mod(rd1, dim, set_tensor, name='resnet2')
    rd2 = tf.nn.dropout(res2, 0.5)
    res3 = resnet_mod(rd2, dim, set_tensor, name='resnet3')
    # lazy implemetation 
    y_out = simple_model(res3, y, set_tensor, numclass=numclass)
    return y_out

