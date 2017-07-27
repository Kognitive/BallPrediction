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

from src.models.RecurrentPredictionModel import RecurrentPredictionModel

class RNADE(RecurrentPredictionModel):

    # Here we build the computation graph for RNADE.
    def __init__(self, D, K, H, C):

        super().__init__("RNADE", K)

        self.C = C

        # get the weight matrix
        W = tf.get_variable("W", [H, D - 1], tf.float32)
        self.x = tf.placeholder(tf.float32, [3, D, C], name="x")

        # get the variables appropriately
        a = list()
        a.append(tf.get_variable("c", [H, 1], tf.float32) @ tf.ones([1, C], tf.float32))

        # calculate the complete a vector
        for d in range(1, D):
            coeff = tf.expand_dims(tf.slice(W, [0, 0], [H, d], 0) @ tf.slice(self.x, [0, 0, 0], [3, d, C]))
            a.append(tf.expand_dims(a[d - 1], 0) + coeff)

        # guide the a values through the hidden layer
        h = list()
        p = tf.constant(0.3)

        # iterate over all elements
        for el in a:
            h.append(tf.nn.relu(p * el))

        # create weights for hidden layer
        V_alpha, b_alpha = RNADE.create_vb_weights("alpha", D, H, K)
        V_mu, b_mu = RNADE.create_vb_weights("mu", D, H, K)
        V_sigma, b_sigma = RNADE.create_vb_weights("sigma", D, H, K)

        # we got now the hidden layer, the next step consists in
        # combining them, so they output the probability given
        # the data
        mixtures = list()
        for d in range(D):

            # infer parameters of gaussians
            alpha = tf.nn.softmax(tf.transpose(V_alpha[d]) @ h[d] + b_alpha[d], dim=0)
            mu = tf.transpose(V_mu[d]) @ h[d] + b_mu[d]
            sigma = tf.exp(tf.transpose(V_sigma[d]) @ h[d] + b_sigma[d])

            # foreach channel of the input data
            dists = list()
            for c in range(C):

                # create mixture of gaussians
                result = tf.constant(1.0, tf.float32)
                for k in range(K):
                    result += alpha[:, c] * self.gaussian(self.x[d, c], mu[k, c], sigma[k, c])

                # save the result
                dists.append(result)

            # stack them up
            dists = tf.stack(dists)

            # add it to the list of mixtures
            mixtures.append(dists)

        # stack up the mixtures
        mixtures = tf.stack(mixtures, axis=0)
        products = tf.reduce_prod(mixtures, axis=0)
        mean = tf.reduce_mean(products, axis=0)

        # create the negative log likelihood
        self.nll = - tf.log(mean)
        self.trainer = tf.train.AdamOptimizer().minimize(self.nll)

        # init the global variables initializer
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    # define the gaussian value
    def gaussian(self, x, mu, sigma):
        root = tf.sqrt(tf.constant(6.28, tf.float32) * tf.pow(sigma, 2))
        normalize = tf.constant(1.0, tf.float32) / root

        offset = tf.pow(x - mu, 2)
        exp_part = tf.exp(-0.5 * offset / tf.pow(sigma, 2))

        return normalize * exp_part

    # This method creates the weights for one layer.
    @staticmethod
    def create_vb_weights(name, D, H, K):

        # create V
        V = list()
        b = list()
        for d in range(D):
            V.append(tf.get_variable("V_" + str(name) + "_" + str(d), [H, K], tf.float32))
            b.append(tf.get_variable("b_" + str(name) + "_" + str(d), [K, 1], tf.float32))

        # create b
        return V, b

    # This method retrieves a list of trajectories. It can further
    # process or transform these trajectories. But the desired overall
    # behaviour is, that it should use the passed trajectories as training
    # data and thus adjust the parameters of the model appropriately.
    #
    # trajectories - This is a list of trajectories (A trajectory is a numpy vector)
    # steps - The number of steps the model should execute.
    #
    def train(self, trajectories, steps):

        # sample the randomly
        slices = np.random.randint(0, np.size(trajectories, 2), self.C)
        traj = trajectories[0, :, slices]

        # we want to perform n steps
        for k in range(steps):
            self.sess.run(self.trainer, feed_dict={self.x: np.transpose(traj)})

    # This method gets a the current state, and tries to output its prediction.
    #
    # - trajectories This is basically one state, e.g. a x-y-z position
    #
    def validate(self, trajectories):
        return self.sess.run(self.nll, feed_dict={self.x: trajectories[0, :, :]})

    # This method has to be implemented, and the desired behaviour consists in resetting all
    # model internal parameters to
    def init_params(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)