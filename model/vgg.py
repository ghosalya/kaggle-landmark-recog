'''
Implementation of VGG-19
'''
import tensorflow as tf 

def conv_layer(intensor, name, set_tensor, prev_kernel=64, kernel=64):
    Wconv1 = tf.get_variable("Wconv{}".format(name), shape=[3, 3, prev_kernel, kernel])
    bconv1 = tf.get_variable("bconv{}".format(name), shape=[kernel])
    set_tensor('Wconv{}'.format(name),Wconv1)
    # layer 1
    # isit 1 stride?
    a1 = tf.nn.conv2d(intensor, Wconv1, strides=[1,1,1,1], padding='SAME') + bconv1
    h1 = tf.nn.relu(a1)
    # d1 = tf.nn.dropout(h1, 0.5)
    return h1

def vgg19_model(X, y, set_tensor, numclass=14951, size=128):
    '''
    VGG19 Model Implementation in tensorflow
    Ax:
    	conv->relu (64)
            ->conv->relu->maxpool (64)
                ->conv->relu (128)
                    ->conv->relu->maxpool (128)
                        ->conv->relu->conv->relu (256)
                            ->conv->relu->maxpool (256)
                                ->conv->relu->conv->relu (512)
                                    ->conv->relu->maxpool (512)
                                        ->conv->relu->conv->relu (512)
                                            ->conv->relu->maxpool (512)
                                                ->affine->affine (4096)


    '''
    flat_dim = size
    # setup variables
    a1 = conv_layer(X, '1', set_tensor, prev_kernel=3, kernel=64)
    a2 = conv_layer(a1, '2', set_tensor, prev_kernel=64, kernel=64)
    ma2 = tf.nn.max_pool(a2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    flat_dim = flat_dim // 2
    ab1 = conv_layer(ma2, '3', set_tensor, prev_kernel=64, kernel=128)
    ab2 = conv_layer(ab1, '4', set_tensor, prev_kernel=128, kernel=128)
    mab2 = tf.nn.max_pool(ab2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    flat_dim = flat_dim // 2
    ac1 = conv_layer(mab2, '5', set_tensor, prev_kernel=128, kernel=256)
    ac2 = conv_layer(ac1, '6', set_tensor, prev_kernel=256, kernel=256)
    ac3 = conv_layer(ac2, '7', set_tensor, prev_kernel=256, kernel=256)
    mac2 = tf.nn.max_pool(ac3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    flat_dim = flat_dim // 2
    ad1 = conv_layer(mac2, '8', set_tensor, prev_kernel=256, kernel=512)
    ad2 = conv_layer(ad1, '9', set_tensor, prev_kernel=512, kernel=512)
    ad3 = conv_layer(ad2, '10', set_tensor, prev_kernel=512, kernel=512)
    mad2 = tf.nn.max_pool(ad3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    flat_dim = flat_dim // 2

    ### this last bit is removed because
    ### size 128 is too small for such a deep network

    # ae1 = conv_layer(mad2, '11', set_tensor, prev_kernel=512, kernel=512)
    # ae2 = conv_layer(ae1, '12', set_tensor, prev_kernel=512, kernel=512)
    # ae3 = conv_layer(ae2, '13', set_tensor, prev_kernel=512, kernel=512)
    # mae2 = tf.nn.max_pool(ae3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    # flat_dim = flat_dim // 2
    mae2 = mad2

    # only max pooling affect size
    # flat_dim = math.ceil((((size - 6) / 3) - 6) / 2)
    flat_size = flat_dim**2 * 512 # 5148

    W1 = tf.get_variable("W1", shape=[flat_size, 4096])
    b1 = tf.get_variable("b1", shape=[4096])
    set_tensor('W1', W1)
    mae2_flat = tf.reshape(mae2,[-1,flat_size])
    affin = tf.matmul(mae2_flat,W1) + b1
    raff = tf.nn.relu(affin)

    W2 = tf.get_variable("W2", shape=[4096, 4096])
    b2 = tf.get_variable("b2", shape=[4096])
    set_tensor('W2', W2)
    # mae2_flat = tf.reshape(mae2,[-1,flat_size])
    affin2 = tf.matmul(raff,W2) + b2
    raff2 = tf.nn.relu(affin2)

    W3 = tf.get_variable("W3", shape=[4096, numclass])
    b3 = tf.get_variable("b3", shape=[numclass])
    set_tensor('W3', W3)
    # h2_flat = tf.reshape(d2,[-1,5184])
    y_out = tf.matmul(raff2,W3) + b3
    return y_out