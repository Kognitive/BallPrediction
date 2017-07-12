# MIT License
#
# Copyright (c) 2017 Markus Semmler, Stefan Fabian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf

# import the necessary packages
from src.data_transformer.concrete.FeedForwardDataTransformer import FeedForwardDataTransformer
from src.models.RecurrentPredictionModel import RecurrentPredictionModel

# Here we define a basic LSTM network with forget gates, that
# can be used to make predictions about the time series in the
# trajectories
class LSTM(RecurrentPredictionModel):

    # This constructs a new LSTM. You have to supply the input size,
    # the size of the hidden or state layer and how often it should
    # be unrolled, e.g. the cropped trajectory size.
    #
    # - I: The size of the input
    # - N: The number of unrolls
    #
    def __init__(self, I, N, K):

        # make ther super call here
        super().__init__("LSTM", 1)

        # save hyper parameters
        self.I = I
        self.N = N

        # create the initial states
        self.ox = tf.placeholder(tf.float32, [I, N+1, K], name="input_state")

        # split up the input
        self.target_h = tf.slice(self.ox, [0, 1, 0], [I, N, K])
        self.x = tf.slice(self.ox, [0, 0, 0], [I, N, K])
        self.C = tf.zeros([I, 1], tf.float32)

        # unstack the input
        all_x = tf.unstack(self.x, axis=1)
        h = all_x[0]
        C = self.C

        # create weights
        weight_list = LSTM.create_weights_for_multiple_layers(I, N)
        hidden_state_list = list()

        # unfold the cell
        for x_in in all_x:

            # create a cell
            h, C = LSTM.create_lstm_cell(weight_list, x_in, h, C)
            hidden_state_list.append(h)

        # the final states
        self.h = tf.stack(hidden_state_list, axis=1)

        # So far we have got the model
        self.error = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(self.target_h - self.h, 2), axis=0))
        self.minimizer = tf.train.AdamOptimizer().minimize(self.error)

        # here comes the step model
        self.step_h = tf.placeholder(tf.float32, [I, 1], name="h")
        self.step_x = tf.placeholder(tf.float32, [I, 1], name="x")
        self.step_C = tf.placeholder(tf.float32, [I, 1], name="C")

        # the model
        self.step_out_h, self.step_out_C = LSTM.create_lstm_cell(weight_list, self.step_x, self.step_h, self.step_C)

        # state vars
        self.current_h = np.zeros([I, 1])
        self.current_C = np.zeros([I, 1])

        # init the global variables initializer
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # This method can be used to initialize a single weight according to
    # its shape. In addition you have to give the weight a name, so it
    # can be matched inside of the graph.
    #
    # - shape: The shape of the variable to create
    # - name: The name of the variable to create
    #
    @staticmethod
    def create_single_weight(shape, name):
        return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.005), name=name)

    # This method can be used to create the weights for one fully-connected layer.
    #
    # - I: The size of the input
    # - C: The unique number of the layer
    #
    @staticmethod
    def create_weights_for_layer(I, C):

        # define the names of the weights
        name_W = "W" + str(C)
        name_R = "R" + str(C)
        name_p = "p" + str(C)
        name_b = "b" + str(C)

        # sample weights in the appropriate sizes
        W = LSTM.create_single_weight([I, I], name_W)
        R = LSTM.create_single_weight([I, I], name_R)
        p = LSTM.create_single_weight([I, 1], name_p)
        b = LSTM.create_single_weight([I, 1], name_b)

        # and pass them back
        return [W, R, p, b]

    # This method creates weights for multiple layers and
    # combines them in one list.
    #
    # - I: The size of the input
    # - O: The size of the output
    # - N: The numbers of layers to create
    #
    @staticmethod
    def create_weights_for_multiple_layers(I, N):

        # iterate N-times and each time append one weight
        # pair to the result list
        all_weights = list()
        for k in range(N):
            all_weights.append(LSTM.create_weights_for_layer(I, k))

        return all_weights

    # This method creates weights for a specific number of layers. In
    # addition you have to supply it with the number of parameters.
    @staticmethod
    def create_layer(activation, weights, x, h, C=0):
        [W, R, p, b] = weights
        if C != 0: return activation(W @ x + R @ h + np.multiply(p, C) + b)
        else: return activation(W @ x + R @ h + b)

    # This method creates a LSTM cell, using the supplied input and
    # hidden state. It basically integrates the standard LSTM
    # architecture with forget gates.
    @staticmethod
    def create_lstm_cell(weight_list, x, h, C):

        # create all gate layers
        forget_gate = LSTM.create_layer(tf.sigmoid, weight_list[0], x, h, C)
        input_gate = LSTM.create_layer(tf.sigmoid, weight_list[1], x, h, C)
        input_data_gate = LSTM.create_layer(tf.tanh, weight_list[2], x, h)

        # update input gate
        input_gate = tf.multiply(input_gate, input_data_gate)

        # now we build up the complete cell
        forgotten_memory = tf.multiply(forget_gate, C)
        new_C = tf.add(input_gate, forgotten_memory)

        # memory gate
        memory_gate = LSTM.create_layer(tf.sigmoid, weight_list[3], x, h, new_C)

        # now we can construct the outputs for this cell
        new_h = tf.multiply(memory_gate, LSTM.lrelu_activation(new_C))

        # pass back both states
        return new_h, new_C

    # This method creates a lrelu activation function.
    @staticmethod
    def lrelu_activation(x):
        return tf.maximum(x, 0.1 * x)

    # This method defines the current h as the first position
    def init_step(self, x):
        self.current_h = x

    # This method performs one step using the learned model.
    def step(self, x):
        [self.current_h, self.current_C] = \
            self.sess.run([self.step_out_h, self.step_out_C],
                          feed_dict={self.step_x: x, self.step_h: self.current_h, self.step_C: self.current_C})

        return self.current_h

    # This method retrieves a list of trajectories. It can further
    # process or transform these trajectories. But the desired overall
    # behaviour is, that it should use the passed trajectories as training
    # data and thus adjust the parameters of the model appropriately.
    #
    # trajectories - This is a list of trajectories (A trajectory is a numpy vector)
    # steps - The number of steps the model should execute.
    #
    def train(self, trajectories, steps):

        # we want to perform n steps
        for k in range(steps):
            self.sess.run(self.minimizer, feed_dict={self.ox: trajectories})

    # This method gets a the current state, and tries to output its prediction.
    #
    # - trajectories This is basically one state, e.g. a x-y-z position
    #
    def validate(self, trajectories):

        return self.sess.run(self.error, feed_dict={self.ox: trajectories})

        # # evaluate the error for all trajectories
        # error = 0
        # for trajectory in trajectories:
        #     print("Hs")
        #     # reset the current c
        #     self.current_C = np.zeros([self.I, 1])
        #     prediction = self.current_h
        #
        #     # transposed trajectory
        #     ttrajectory = np.transpose(trajectory)
        #     self.current_h = ttrajectory[:, 0:1]
        #
        #     # execute the steps
        #     for n in range(0, np.size(ttrajectory, 1)):
        #
        #         # add the error for the prediction
        #         x = ttrajectory[:, n:n+1]
        #         diff = prediction - x
        #         error += np.outer(diff, diff)
        #
        #         # one step in the model
        #         prediction = self.step(x)


    # This method sh
    def init_params(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)