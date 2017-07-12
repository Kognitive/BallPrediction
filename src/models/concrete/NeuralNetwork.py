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

import tensorflow as tf

from src.models.FeedForwardPredictionModel import FeedForwardPredictionModel

# this class represents a neural network model in our learning framework.


class NeuralNetwork(FeedForwardPredictionModel):

    # this represents a basic fully connected neural network
    def __init__(self, structure, I, O, K):

        # create input and prevsize
        super().__init__("Basic Neural Network", I, O, K)
        prevsize = structure[0]
        self.x = tf.placeholder(tf.float32, shape=[prevsize, None])

        # create a tree
        prevtree = self.x
        tree = self.x

        # iterate over structure
        for k in range(1, len(structure)):

            # get nextsize as well
            nextsize = structure[k]
            weightindex = str(k - 1)
            b = self.init_single_weight([nextsize, 1], "b" + weightindex, True)
            W = self.init_single_weight([nextsize, prevsize], "W" + weightindex)

            # next layer
            prevsize = nextsize
            prevtree = W @ tree + b
            tree = tf.maximum(prevtree, 0.1 * prevtree)

        # the evaluation model
        self.eval_model = prevtree

        # now create the training graph
        self.y = tf.placeholder(tf.float32, shape=[prevsize, None])
        self.error = 0.5 * tf.reduce_sum(tf.reduce_sum((self.eval_model - self.y) ** 2, axis=1), axis=0)

        # define the trainer
        self.trainer = tf.train.AdamOptimizer().minimize(self.error)

        # init the global variables initializer
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # this method returns a weight matrix with the given shape
    def init_single_weight(self, shape, name, bias=False):
        if bias: return tf.Variable(tf.zeros(shape), name=name)
        return tf.Variable(tf.random_normal(shape, mean=0.0, stddev=0.01), name=name)

    # this method should be implemented to provide a train method
    #
    # - inputs: The inputs for the model, based on which it should
    #           predict.
    # - predictions: The predictions, such that an error can be
    #                calculated
    # steps - The number of steps the model should execute.
    #
    def fftrain(self, inputs, predictions, steps):
        for num_step in range(steps):
            self.sess.run(self.trainer, feed_dict={ self.x: inputs, self.y: predictions })

    # this method should be used to predict the next state based
    # on the current inputs
    #
    # - inputs just the list of frames
    def ffpredict(self, inputs):
        return self.sess.run(self.eval_model, feed_dict={ self.x: inputs })

    def init_params(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)