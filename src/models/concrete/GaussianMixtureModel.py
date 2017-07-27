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

class GaussianMixtureModel:

    # Here we create the initial Gaussian Mixture Model used for Prediction
    def __init__(self, I, NH, K):

        # define the two placeholder for the model
        x_pred = tf.placeholder(tf.float32, [I, None], name="x_pred")
        x_hist = tf.placeholder(tf.float32, [I, NH, None], name="x_hist")

        # create the result, which is initially zero
        model = tf.constant(0.0, tf.float32)

        # repeat for K times
        for k in range(K):
            pi_k = tf.get_variable("pi" + str(k), [1], tf.float32)
            mu_k = tf.get_variable("mu" + str(k), [NH + 1], tf.float32)
            sigma_k = tf.get_variable("sigma" + str(k), [NH + 1, NH + 1], tf.float32)
            gauss = tf.contrib.distributions.MultivariateNormalDiag(mu_k, sigma_k)
            model *= pi_k * gauss

        self.nll = -tf.log(model)
        self.trainer = tf.train.AdamOptimizer(self.nll).minimize()

    # This method retrieves a list of trajectories. It can further
    # process or transform these trajectories. But the desired overall
    # behaviour is, that it should use the passed trajectories as training
    # data and thus adjust the parameters of the model appropriately.
    #
    # trajectories - This is a list of trajectories (A trajectory is a numpy vector)
    # steps - The number of steps the model should execute.
    #
    def train(self, trajectories, steps):
        raise NotImplementedError("Please implement a training method for " + str(self.name))

    # This method gets a the current state, and tries to output its prediction.
    #
    # - trajectories This is basically one state, e.g. a x-y-z position
    #
    def validate(self, trajectories):
        raise NotImplementedError("Please implement a step method for " + str(self.name))

    # This method has to be implemented, and the desired behaviour consists in resetting all
    # model internal parameters to
    def init_params(self):
        raise NotImplementedError("Please implement a init parameters method for " + str(self.name))