'''
Simple model implementation
for default purposes
'''
import tensorflow as tf 

def simple_model(X, y, set_tensor, numclass=14951):
    '''
    Simple Model Implementation in tensorflow
    Ax:
    	conv->relu->dropout->conv->relu->dropout->affine
    '''

    # setup variables
    Wconv1 = tf.get_variable("Wconv1", shape=[7, 7, 3, 32])
    bconv1 = tf.get_variable("bconv1", shape=[32])
    set_tensor(Wconv1)
    Wconv2 = tf.get_variable("Wconv2", shape=[7, 7, 32, 16])
    bconv2 = tf.get_variable("bconv2", shape=[16])
    set_tensor(Wconv2)
    W1 = tf.get_variable("W1", shape=[13456, numclass])
    b1 = tf.get_variable("b1", shape=[numclass])
    set_tensor(W1)

    # define our graph 
    a1 = tf.nn.conv2d(X, Wconv1, strides=[1,3,3,1], padding='VALID') + bconv1
    h1 = tf.nn.relu(a1)
    d1 = tf.nn.dropout(h1, 0.5)
    a2 = tf.nn.conv2d(d1, Wconv2, strides=[1,2,2,1], padding='VALID') + bconv2
    h2 = tf.nn.relu(a2)
    d2 = tf.nn.dropout(h2, 0.5)
    h2_flat = tf.reshape(d2,[-1,13456])
    y_out = tf.matmul(h2_flat,W1) + b1
    return y_out