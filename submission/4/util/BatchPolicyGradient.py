import numpy as np
import pandas as pd
import os
from util.CustomLogger import CustomLogger
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
            loss = - K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred), axis=-1)
            return loss

        self.environment = environment
        self.decimal = decimal
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.logger = logger

        #self.episodes = []
        self.y_hist = []
        self.x_hist = []
        self.best_policy = np.round(np.random.rand(2,), self.decimal)
        self.best_predict = []
        self.best_policy_reward = []
        self.best_policy_hist = []
        self.best_policy_reward_hist = []

        x_range = np.linspace(0, 1, 10**min(self.decimal,4) + 1)[1:]
        self.x_discrete = np.array(list(itertools.permutations(x_range, 2)))

        self.model = Sequential()
        self.model.add(Dense(4, input_dim=2, activation='relu'))
        self.model.add(Dense(2, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(loss=log_loss, optimizer='adam', metrics=['mse','mae'])

    def sample_policy(self, p):
        if p < self.epsilon:
            policy = np.array([self.best_policy])
        else:
            policy = np.round(np.random.rand(1,2), self.decimal)
        return policy

    def sample_batch_policy(self):
        """
        :return: 2d ndarray
        """
        batch_policy = np.squeeze(list(map(self.sample_policy, np.random.rand(self.batch_size))))
        return batch_policy

    def find_best_policy(self):
        # save best policy so far
        best_predict = self.model.predict(self.x_discrete)
        self.best_policy = self.x_discrete[np.argmax(best_predict)]
        self.best_predict.append(best_predict.max())
        self.best_policy_reward = self.environment.evaluateReward(self.best_policy)

        self.best_policy_hist.append(self.best_policy)
        self.best_policy_reward_hist.append(self.best_policy_reward)


    def train(self, epochs):

        for i in range(epochs):
            # pick policy
            x = self.sample_batch_policy()

            # get results
            y = np.array(self.environment.evaluateReward(x))
            norm_y = (y - y.min()) / (y.max() - y.min())

            self.x_hist.append(x)
            self.y_hist.append(y)

            # train
            self.model.fit(x, norm_y, epochs=1, batch_size=len(x), verbose=2)

            # find best policy
            self.find_best_policy()

        pd.DataFrame(self.best_policy_hist).to_csv('best_policy_hist.csv', index=False)
        pd.DataFrame(self.best_policy_reward_hist).to_csv('best_policy_reward_hist.csv', index=False)
        pd.DataFrame(self.best_predict).to_csv('best_predict.csv', index=False)
        np.array(self.x_hist).dump('x_hist')
        np.array(self.y_hist).dump('y_hist')

    def create_submissions(self, filename='my_submission.csv'):
        if os.path.exists(filename):
            raise Exception('output csv {} already exists'.format(filename))

        data = {
            'episode_no': [i for i in range(len(self.best_policy_reward_hist))],
            'rewards': np.array(self.best_policy_reward_hist),
            'policy': self.best_policy_hist,
        }
        submission_file = pd.DataFrame(data)
        submission_file.to_csv(filename, index=False)