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


class RecurrentNeuralNetwork(RecurrentPredictionModel):
    """This model represents a standard recurrent neural network."""

    def __init__(self, config):
        """Constructs a new RNN.

        Args:
            config: The configuration parameters
                unique_name: Define the unique name of this lstm
                num_input: The number of input units per step.
                num_output: The number of output units per step.
                num_hidden: The number of units in the hidden layer.
                num_cells: The number of cells per layer
                num_layers: Define number of time-step unfolds.
                clip_norm: The norm, to which a gradient should be clipped
                batch_size: This represents the batch size used for training.
                minimizer: Select the appropriate minimizer
                seed: Represents the seed for this model
                momentum: The momentum if the minimizer is momentum
                lr_rate: The initial learning rate
                lr_decay_steps: The steps until a decay should happen
                lr_decay_rate: How much should the learning rate be reduced
        """

        # Perform the super call
        self.config = config
        config['unique_name'] = "RNN_" + config['unique_name']
        super().__init__(config['unique_name'])

        # save hyper parameters
        self.weights_initializer = tf.contrib.layers.variance_scaling_initializer()
        self.bias_initializer = tf.constant_initializer(0.0)
        # tf.random_normal_initializer(0.0, 0.1, seed=config['seed'])

        with tf.variable_scope(config['unique_name']):

            # init
            self.init_common()

            # initialize num_cells different cells
            for i in range(self.config['num_cells']):
                self.init_cell(str(i))

            # --------------------------- TRAINING ----------------------------

            self.current_step = -1
            self.step_num = tf.placeholder(tf.int32, [], name="step_num")
            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.learning_rate = tf.train.exponential_decay(
                self.config['lr_rate'],
                self.global_step,
                self.config['lr_decay_steps'],
                self.config['lr_decay_rate'],
                staircase=False)

            # create a tensor for the input
            self.x = tf.placeholder(tf.float32, [config['num_input'], config['num_layers'], None], name="input")
            self.y = tf.placeholder(tf.float32, [config['num_input'], None], name="target")

            # define the memory state
            self.h = self.get_h()

            # unstack the input
            unstacked_x = tf.unstack(self.x, axis=1)

            # use for dynamic
            h = self.get_initial_h()

            # unfold the cell
            for x_in in unstacked_x:

                # create a cell
                self.target_y, h = self.create_combined_cell(x_in, h)

            # first of create the reduced squared error
            red_squared_err = tf.reduce_sum(tf.pow(self.target_y - self.y, 2), axis=0)

            # So far we have got the model
            self.error = 0.5 * tf.reduce_mean(red_squared_err, axis=0)
            self.a_error = tf.reduce_mean(tf.sqrt(red_squared_err), axis=0)
            self.minimizer = self.create_minimizer(self.learning_rate, self.error, self.global_step)

            # ------------------------------ EVALUATION ---------------------------------

            # here comes the step model
            self.step_x = tf.placeholder(tf.float32, [config['num_input'], 1], name="step_x")
            self.step_h = self.get_step_h()

            # the model
            self.step_y, self.step_out_h = self.create_combined_cell(self.step_x, self.step_h)

        # state vars
        self.current_h = self.get_current_h()

        # init the global variables initializer
        tf.set_random_seed(self.config['seed'])
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_h(self):
        """Gets a reference to the step h."""
        raise NotImplementedError("Get the current h")

    def get_initial_h(self):
        """Gets a reference to the step h."""
        raise NotImplementedError("Get the current initial h")

    def get_step_h(self):
        """Retrieve the step h"""
        raise NotImplementedError("Retrieve the step hidden state.")

    def get_current_h(self):
        """Deliver current h"""
        raise NotImplementedError("Retrieve the current hidden state.")

    def init_common(self):
        """This initializes the common variables.
        """
        with tf.variable_scope("common", reuse=None, ):
            tf.get_variable("O", [self.config['num_input'], self.config['num_hidden'] * self.config['num_cells']],
                            dtype=tf.float32, initializer=self.weights_initializer)

    def create_minimizer(self, learning_rate, error, global_step):
        """This method creates the correct optimizer."""

        # first of all set the global step to zero, because we train from the beginning
        minimizer = 0

        # choose between the available optimizers
        str_minimizer = self.config['minimizer']
        if str_minimizer == 'momentum':
            minimizer = tf.train.MomentumOptimizer(learning_rate, self.config['momentum'], use_nesterov=True)

        elif str_minimizer == 'adam':
            minimizer = tf.train.AdamOptimizer(learning_rate)

        elif str_minimizer == 'rmsprop':
            minimizer = tf.train.RMSPropOptimizer(learning_rate)

        else:
            print('Unknown minimizer ' + str_minimizer)
            exit(1)

        # calculate the gradients
        gradients = minimizer.compute_gradients(error)

        # when clipping should be performed
        if self.config['clip_norm'] > 0:
            gradients, variables = zip(*gradients)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=self.config['clip_norm'])

            # define the trainer
            trainer = minimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step)

        else:
            trainer = minimizer.apply_gradients(gradients, global_step=global_step)

        return trainer

    def init_cell(self, name):
        """This method should initialize one cell.

        Args:
            name: The name of the cell.
        """
        raise NotImplementedError("Please implement init_cell")

    def create_cell(self, name, x, h, num_cell):
        """This method creates a RNN cell. It basically uses the
        previously initialized weights.

        [h, s] = h_state
            new_h: The new hidden vector
        """
        raise NotImplementedError("Please implement create_cell")

    def create_combined_cell(self, x, h):
        """This method creates a combined rnn cell. It basically
        connects them appropriately.

        Args:
            x: The input vector to the cell (self.I)
            h: The hidden vector (self.I * self.num_cells)

        Returns:
            Tuple(y, h), where y is the predicted output,
            h is the hidden vector for all cells
        """

        # create all cells
        all_length = len(h)
        all_new_h = list()
        [all_new_h.append(list()) for k in range(all_length)]

        for k in range(self.config['num_cells']):
            new_h = self.create_cell(str(k), x, h, k)

            # append it to each list
            for l in range(all_length):
                all_new_h[l].append(new_h[l])

        # concat all h
        concat_new_h = [tf.concat(comb_new_h, 0) for comb_new_h in all_new_h]

        # create the output
        with tf.variable_scope("common", reuse=True):
            O = tf.get_variable("O")
            y = O @ concat_new_h[0]

        # pass back the cell outputs
        return y, concat_new_h

    @staticmethod
    def lrelu_activation(x):
        """This method creates a leaky relu function.

        Args:
            x: The input to the layer

        Returns:
            The layer after the activation is applied.
        """

        return tf.maximum(x, tf.constant(0.01) * x)

    def init_step(self, x):

        """This method is required to set the initial hidden state."""
        self.current_h = x

    def step(self, x):
        """This method can be used to perform a step on the model.

        Args:
            x: The input at the current step.

        Returns:
            The result obtained from exploiting the inner model.
        """

        dic = dict(zip(self.step_h, self.current_h))
        dic.update({self.step_x: x})
        res = self.sess.run(self.step_y, dic)
        self.current_h = self.sess.run(self.step_out_h, dic)

        return res

    def train(self, trajectories, target_trajectories, steps):
        """This method retrieves a list of trajectories. It can further
        process or transform these trajectories. But the desired overall
        behaviour is, that it should use the passed trajectories as training
        data and thus adjust the parameters of the model appropriately.

        Args:
            trajectories: This is a list of trajectories (A trajectory is a numpy vector)
            target_trajectories: The target trajectories
            steps: The number of steps the model should execute.
        """

        self.current_step += 1

        # we want to perform n steps
        for k in range(steps):
            self.sess.run(self.minimizer, feed_dict={self.x: trajectories, self.y: target_trajectories, self.step_num: self.current_step})

    def validate(self, trajectories, target_trajectories):
        """This method basically validates the passed trajectories.
        It therefore splits them up so that the future frame get passed as the target.

        Args:
            trajectories: The trajectories to use for validation.
            target_trajectories: The target trajectories

        Returns:
            The error on the passed trajectories
        """

        self.current_step += 1
        return self.sess.run(self.a_error, feed_dict={self.x: trajectories, self.y: target_trajectories, self.step_num: self.current_step})

    def init_params(self):
        """This initializes the parameters."""
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def reset(self):
        tf.reset_default_graph()
