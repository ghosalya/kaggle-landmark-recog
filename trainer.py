'''
Script of the Tensorflow Trainer class 
This should not be modified if you are
modifying architecture
'''

import tensorflow as tf 
import numpy as np 
import math, pickle, random
import matplotlib.pyplot as plt
from datetime import datetime

from utils import *
# importing models 
from model.simple import simple_model
from model.fakealex import fakealex_model
from model.resnet import resnet_model
from model.glr import glr_model
from model.vgg import vgg19_model

# NUM_CLASSES = 14951
NUM_CLASSES = 1323 # reflect the train160.csv
IMG_SIZE = 500

class Trainer():
    def __init__(self, model=simple_model):
        '''
        Constructor takes model which is a function
        model must take args: X, y
        and return a tensorflow network
        '''
        self.model = model
        self.data_len = None
        # control over dataset slices
        # to use as training or valdiation
        self.train_slice = []
        self.val_slice = []

    def get_model(self, X, y):
        return self.model(X, y, 
                          self.set_tensorboard_var, 
                          numclass=NUM_CLASSES,
                          size=IMG_SIZE)

    def set_tensorboard_var(self, name, var):
        '''
        Sets various tensorflow parameters
        for the tensorboard interface
        '''
        with tf.name_scope(name):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def sgd_optimizer(self, loss, learning_rate=5e-4):
        # define SGD optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
        train_step = optimizer.minimize(loss)
        return optimizer, train_step

    def adam_optimizer(self, loss, learning_rate=1.5e-4):
        # define Adam Optimizer - Seems to be much better
        optimizer = tf.train.AdamOptimizer(learning_rate,
                                            beta1=0.8,
                                            beta2=0.988) # faster decay 
        train_step = optimizer.minimize(loss)
        return optimizer, train_step

    def load_model(self, sess, filename):
        # setup placeholder
        self.tfX = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
        self.tfy = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

        # get tensorflow model
        # saver = tf.train.Saver(var_list=[])
        self.y_out = self.get_model(self.tfX, self.tfy)
        saver = tf.train.Saver()
        saver.restore(sess, filename)
        print(':==> Loading')

    def setup_slices(self):
        '''
        Setup a list of indices
        to use as train/val slice
        '''
        print('setting up slices')
        datalen = get_dataset_length()
        train_size = int(datalen * 4 / 5)
        val_size = datalen - train_size

        print('assigning slices')
        self.data_len = datalen
        print('  ==> train')
        self.train_slice = random.sample(range(datalen), train_size)
        print('  ==> val')
        self.train_slice = random.sample(range(datalen), val_size) #this implementation doesnt exclude

        ## this implementation takes too long
        # self.val_slice = [i for i in range(datalen)
        #                     if i not in self.train_slice]
        print('done')

    def run_training(self, sess, overfit=False):
        '''
        Runs model on training mode
        on the given tensorflow session
        '''

        # setup placeholder
        self.tfX = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 3])
        self.tfy = tf.placeholder(tf.int64, [None])
        self.is_training = tf.placeholder(tf.bool)

        # get tensorflow model
        self.y_out = self.get_model(self.tfX, self.tfy)

        # define our loss
        total_loss = tf.losses.hinge_loss(tf.one_hot(self.tfy,NUM_CLASSES),
                                          logits=self.y_out)
        self.mean_loss = tf.reduce_mean(total_loss)
        optimizer, train_step = self.adam_optimizer(self.mean_loss)
        train_writer = tf.summary.FileWriter('logs/train', graph=tf.get_default_graph())

        # run the model
        return self.run_model(sess, self.y_out, self.mean_loss,
                       epochs=1, batch_size=250,training=train_step, 
                       tensorboard_writer=train_writer, 
                       # iterr=100,
                       overfit=overfit)

    def run_validation(self, sess):
        '''
        Runs model on validation mode (without backprop)
        indicated by not having train_step
        '''
        # Xd, yd = get_dataset(batch=2000)
        return self.run_model(sess, self.y_out, self.mean_loss, #Xd=Xd, yd=yd,
                       epochs=1, batch_size=250, iterr=1000)

    def run_model(self, session, predict, loss_val, Xd=None, yd=None,
              epochs=1, batch_size=64, iterr=0,training=None, 
              tensorboard_writer=None, overfit=False):
        '''
        Run the model through tensorflow
        - if training, give training=train_step, and dont give Xd & yd
        - if validation, give Xd and yd, dont give training
        '''
        correct_prediction = tf.equal(tf.argmax(predict,1), self.tfy)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        training_now = (training is not None)

        # tensorboard stuff
        tf.summary.scalar("cost", loss_val)
        tf.summary.scalar("accuracy", accuracy)
        summary_op = tf.summary.merge_all()
        
        session.run(tf.global_variables_initializer())
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        if training_now:
            variables = [loss_val, correct_prediction, summary_op, training]
            saver = tf.train.Saver()
        else:
            variables = [loss_val, correct_prediction, accuracy]

        if self.data_len is None:
            self.data_len = get_dataset_length()
            print_tele('running through data of length {}'.format(self.data_len))

        if iterr == 0:
            # if not given, assume run through the whole dataset
            iterr = math.ceil(self.data_len*4/(batch_size*5))

        # counter 
        iter_cnt = 0
        # keep track of losses
        losses = []
        for e in range(epochs):
            # keep track of accuracy
            correct = 0
            # set training & validation slices
            self.setup_slices()

            # make sure we iterate over the dataset once
            total_data = 0
            for i in range(iterr):
                if Xd is None:
                    if training_now:
                        # _Xd, _yd = get_dataset(batch=batch_size)
                        idx = self.train_slice[i*batch_size: min((i+1)*batch_size,
                                                                 len(self.train_slice))]
                        _Xd, _yd = get_dataset(index=idx)
                        if overfit:
                            Xd = _Xd
                            yd = _yd
                    else:
                        _Xd, _yd = get_dataset(batch=batch_size, 
                            index=i*batch_size)
                else:
                    _Xd = Xd
                    _yd = yd



                feed_dict = {self.tfX: _Xd,
                             self.tfy: _yd,
                             self.is_training: training_now }
                # get batch size
                actual_batch_size = _yd.shape[0] 
                total_data += _Xd.shape[0]
                
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                if training_now:
                    loss, corr, summary, _ = session.run(variables,feed_dict=feed_dict)
                    tensorboard_writer.add_summary(summary, e * iterr + i )
                else:
                    loss, corr, _ = session.run(variables,feed_dict=feed_dict)

                corr = np.array(corr).astype(np.float32)
                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
                
                # print every now and then
                # if training_now and (iter_cnt % print_every) == 0:
                if iter_cnt % 50 == 0 and training_now:
                    print_tele("Iteration {0} of {3}: with minibatch training loss = {1:.3g} and accuracy of {2:.4g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size, iterr))
                elif training_now:
                    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.4g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                elif iter_cnt % 50 == 0:
                    print_tele("Validation Iter {0}: with minibatch accuracy of {1:.4g}"\
                      .format(iter_cnt,np.sum(corr)/actual_batch_size))
                iter_cnt += 1

                if loss == 0:
                    print_tele('reaching 0 loss, breaking iteration')
                    break
                
            total_correct = correct/total_data
            total_loss = np.sum(losses)/total_data
            print_tele("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            if training_now:
                savename = "./saved_model/{}_{}_{}.tfc"\
                        .format(self.model.__name__, e,
                                datetime.now().strftime("%Y%m%d_%H:%M"))
                saver.save(session, savename, global_step=e*iterr)
            
        if training_now:
            plt.plot(losses)
            plt.grid(True)
            plt.title('Epoch {} Loss'.format(e+1))
            plt.xlabel('minibatch number')
            plt.ylabel('minibatch loss')
            plt.savefig('./graphs/losses_{}_{}.png'.format(self.model.__name__,
                                                  datetime.now().strftime("day%d_%H:%M")))
            # plt.show()
        return total_loss,total_correct,losses


def main():
    # trainer = Trainer(model=resnet_model)
    trainer = Trainer(model=resnet_model)
    # saver = tf.train.Saver()
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                    log_device_placement=True)) as sess:
        # tensorboard cant run on GPU for some reason, so need placements
        with tf.device("/gpu:0"): #change either /gpu or /cpu
            try:
                print_tele('Training')
                train_result = trainer.run_training(sess, overfit=False)
                print("Run the command line:\n" \
                          "--> tensorboard --logdir=logs/train " \
                          "\nThen open http://0.0.0.0:6006/ into "\
                          "your web browser")
                # trainer.load_model(sess, '/home/student04/gede/repo/saved_model/glr_model_6_20180413_02:28.tfc-2400.index')
                print('Validation')
                val_result = trainer.run_validation(sess)
                # trainer.run_model(sess,y_out,mean_loss,X_val,y_val,1,64)
                print_tele("Training complete!")
                print_tele('\n\nTraining result (loss, correct):' + str(train_result[:2]))
                print_tele('Validation result (loss, correct):' + str(val_result[:2]))
            except Exception as e:
                print_tele("training ERROR!")
                print_tele(str(e))
                raise e


if __name__ == '__main__':
    main()