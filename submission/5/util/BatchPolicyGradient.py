import numpy as np
import pandas as pd
import os
from util.CustomLogger import CustomLogger
import keras
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import tensorflow as tf
import itertools

#!pip3 install git+https://github.com/slremy/netsapi --user --upgrade

from sys import exit, exc_info, argv

logger = CustomLogger(__name__)

class BatchPolicyGradient:
    def __init__(self, environment, decimal=2, epsilon=0.1, batch_size=16, logger=None):

        def log_loss(y_true, y_pred):
            # https://stackoverflow.com/questions/49201632/how-to-debug-custom-loss-function-in-keras
            loss = - K.mean(y_true * K.log(K.clip(y_pred, 1e-7, 1-1e-7)) + (1 - y_true) * K.log(K.clip(1 - y_pred, 1e-7, 1-1e-7)), axis=-1)
            return loss

        self.environment = environment
        self.decimal = decimal
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.logger = logger

        #self.episodes = []
        self.y_hist = []
        self.x_hist = []
        self.best_x = np.round(np.random.rand(2, ), self.decimal)
        self.best_y_hat = []
        self.best_y = []
        self.best_x_hist = []
        self.best_y_hist = []

        x_range = np.linspace(0, 1, 10**min(self.decimal,3) + 1)[1:]
        self.x_discrete = np.array(list(itertools.permutations(x_range, 2)))

        self.model = Sequential()
        self.model.add(Dense(4, input_dim=2, activation='relu'))
        self.model.add(Dense(2, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        adam = keras.optimizers.Adam(lr=0.01)
        self.model.compile(loss=log_loss, optimizer=adam, metrics=['mse','mae'])

        self.model = keras.models.load_model('model', custom_objects={'log_loss': log_loss})

    def sample_policy(self, p):
        if p < self.epsilon:
            policy = np.array([self.best_x])
        else:
            policy = np.round(np.random.rand(1,2), self.decimal)
        return policy

    def sample_batch_policy(self):
        """
        :return: 2d ndarray
        """
        batch_policy = np.squeeze(list(map(self.sample_policy, np.random.rand(self.batch_size))))
        return batch_policy

    def find_best_x(self):
        # save best policy so far
        # best_x is ndarray of shape (2,)
        best_predict = self.model.predict(self.x_discrete)
        self.best_x = self.x_discrete[np.argmax(best_predict)]
        self.best_y_hat.append(best_predict.max())
        self.best_y = self.environment.evaluateReward(self.best_x)[:, 0].astype(float)[0]

        self.best_x_hist.append(self.best_x)
        self.best_y_hist.append(self.best_y)

    def new_find_best_policy(self):
        output = self.model.layers[-1].output
        loss = K.mean(output[:, 0])
        grads = K.gradients(loss, self.model.input)[0]  # the output of `gradients` is a list, just take the first (and only) element
        grads = K.l2_normalize(grads)  # normalize the gradients to help having an smooth optimization process

        func = K.function([self.model.input], [loss, grads])

        input_img = np.array([[0.5, 0.5]])
        lr = 0.001  # learning rate used for gradient updates
        max_iter = 50  # number of gradient updates iterations
        for i in range(max_iter):
            loss_val, grads_val = func([input_img])
            input_img += grads_val * lr
            input_img = np.clip(input_img, 1e-7, 1)
        input_img

    def train(self, epochs):

        for i in range(epochs):
            logger.info('================== Training round {} Start =================='.format(i))
            # pick policy
            x = self.sample_batch_policy()

            # get results
            y = self.environment.evaluateReward(x)

            x = x[y[:,0] != 'nan']
            y = y[y[:,0] != 'nan'][:,0].astype(float)

            self.x_hist.append(x)
            self.y_hist.append(y)

            norm_y = (y - np.array(self.y_hist).ravel().min()) / (np.array(self.y_hist).ravel().max() - np.array(self.y_hist).ravel().min())

            # train
            self.model.fit(x, norm_y, epochs=10, batch_size=int(len(x)/5), verbose=2)

            # find best policy
            self.find_best_x()

        self.model.save('model')
        pd.DataFrame(self.best_x_hist).to_csv('best_policy_hist.csv', index=False)
        pd.DataFrame(self.best_y_hist).to_csv('best_policy_reward_hist.csv', index=False)
        pd.DataFrame(self.best_y_hat).to_csv('best_predict.csv', index=False)
        np.save('x_hist', self.x_hist)
        np.save('y_hist', self.y_hist)

    def create_submissions(self, filename='my_submission.csv', _trial=10):
        if os.path.exists(filename):
            raise Exception('output csv {} already exists'.format(filename))

        self.find_best_x()
        rewards = self.environment.evaluateReward(np.array([self.best_x] * _trial), skip_cache=True)
        rewards = rewards[rewards[:, 0] != 'nan'][:, 0].astype(float)

        logger.info('best x = {}'.format(self.best_x))
        logger.info('mean = {}, std = {}, score = {}'.format(np.mean(rewards), np.std(rewards), np.mean(rewards)/(np.std(rewards)+1e-7)))

        data = {
            'episode_no': [i for i in range(_trial)],
            'rewards': np.array(rewards),
            'policy': [self.best_x] * _trial,
        }
        logger.debug(data)
        submission_file = pd.DataFrame(data)
        submission_file.to_csv(filename, index=False)