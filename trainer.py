'''
Script of the Tensorflow Trainer class 
This should not be modified if you are
modifying architecture
'''

import tensorflow as tf 
import numpy as np 
import math

from utils import *
# importing models 
from model.simple import simple_model
from model.fakealex import fakealex_model

NUM_CLASSES = 14951
IMG_SIZE = 196

class Trainer():
    def __init__(self, model=simple_model):
        '''
        Constructor takes model which is a function
        model must take args: X, y
        and return a tensorflow network
        '''
        self.model = model

    def get_model(self, X, y):
        return self.model(X, y, 
                          self.set_tensorboard_var, 
                          numclass=NUM_CLASSES)

    def set_tensorboard_var(self, var):
        '''
        Sets various tensorflow parameters
        for the tensorboard interface
        '''
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def sgd_optimizer(self, loss, learning_rate=2e-4):
        # define SGD optimizer
        optimizer = tf.train.GradientDescentOptimizer(learning_rate) 
        train_step = optimizer.minimize(loss)
        return optimizer, train_step

    def adam_optimizer(self, loss, learning_rate=1e-4):
        # define Adam Optimizer - Seems to be much better
        optimizer = tf.train.AdamOptimizer(learning_rate) 
        train_step = optimizer.minimize(loss)
        return optimizer, train_step

    def run_training(self, sess):
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
        self.run_model(sess, self.y_out, self.mean_loss,
                       epochs=1, batch_size=400, print_every=25,
                       iterr=500, training=train_step, 
                       tensorboard_writer=train_writer)

    def run_validation(self, sess):
        '''
        Runs model on validation mode (without backprop)
        '''
        Xd, yd = get_dataset(batch=2000)
        self.run_model(sess, self.y_out, self.mean_loss, Xd=Xd, yd=yd,
                       epochs=1, batch_size=100, print_every=100)


    def run_model(self, session, predict, loss_val, Xd=None, yd=None,
              epochs=1, batch_size=64, print_every=2, iterr=1,
              training=None, tensorboard_writer=None):
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
        else:
            variables = [loss_val, correct_prediction, accuracy]

        # counter 
        iter_cnt = 0
        # keep track of losses
        losses = []
        for e in range(epochs):
            # keep track of accuracy
            correct = 0
            # make sure we iterate over the dataset once
            total_data = 0
            for i in range(iterr):
                # inner loading
                if Xd is None:
                    _Xd, _yd = get_dataset(batch=batch_size)
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
                    tensorboard_writer.add_summary(summary, e * iterr + 1 )
                else:
                    loss, corr, _ = session.run(variables,feed_dict=feed_dict)

                corr = np.array(corr).astype(np.float32)
                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
                
                # print every now and then
                # if training_now and (iter_cnt % print_every) == 0:
                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.4g}"\
                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
                
            total_correct = correct/total_data
            total_loss = np.sum(losses)/total_data
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
            
        # if plot_losses:
        #     plt.plot(losses)
        #     plt.grid(True)
        #     plt.title('Epoch {} Loss'.format(e+1))
        #     plt.xlabel('minibatch number')
        #     plt.ylabel('minibatch loss')
        #     plt.show()
                
        return total_loss,total_correct,losses


def main():
    trainer = Trainer(model=fakealex_model)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
                    log_device_placement=True)) as sess:
        # tensorboard cant run on GPU for some reason, so need placements
        with tf.device("/gpu:0"): #change either /gpu or /cpu
            print('Training')
            trainer.run_training(sess)
            print("Run the command line:\n" \
                      "--> tensorboard --logdir=logs/train " \
                      "\nThen open http://0.0.0.0:6006/ into "\
                      "your web browser")
            # print('Validation')
            # sfc.run_validation(sess)
            # sfc.run_model(sess,y_out,mean_loss,X_val,y_val,1,64)


if __name__ == '__main__':
    main()