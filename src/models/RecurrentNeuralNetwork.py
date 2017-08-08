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
        self.weights_initializer = tf.contrib.layers.variance_scaling_initializer(1.0, 'FAN_AVG', True, config['seed'])
        self.bias_initializer = tf.constant_initializer(0.0)

        with tf.variable_scope(config['unique_name']) as scope:

            # init some variables
            O = tf.get_variable("O", [self.config['num_input'], self.config['num_hidden']], dtype=tf.float32, initializer=self.weights_initializer)
            self.init_pre_processing_layer()

            # init combined cells
            for k in range(config['num_stacks']):
                self.init_cell(str(k))

            # --------------------------- TRAINING ----------------------------

            self.current_step = 0
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
            self.h = list()
            for k in range(config['num_stacks']):
                self.h.append(self.get_h())

            # unstack the input
            unstacked_x = tf.unstack(self.x, axis=1)

            # use for dynamic
            h = self.h

            # unfold the cell
            for x_in in unstacked_x:

                # create a cell
                processed_x_in = self.create_pre_processing_layer(x_in)

                # stack them up
                for k in range(config['num_stacks']):
                    h[k] = self.create_cell(str(k), processed_x_in, h[k])

            self.target_y = O @ h[config['num_stacks'] - 1][0]

            # first of create the reduced squared error
            red_squared_err = tf.reduce_sum(tf.pow(self.target_y - self.y, 2), axis=0)

            # So far we have got the model
            self.error = 0.5 * tf.reduce_mean(red_squared_err, axis=0)
            self.a_error = tf.reduce_mean(tf.sqrt(red_squared_err), axis=0)
            self.minimizer = self.create_minimizer(self.learning_rate, self.error, self.global_step)

            # ------------------------------ EVALUATION ---------------------------------

            # here comes the step model
            self.step_x = tf.placeholder(tf.float32, [config['num_input'], 1], name="step_x")
            step_h = list()
            for k in range(config['num_stacks']):
                step_h.append(self.get_step_h())

            # the model
            step_pre_processed_x_in = self.create_pre_processing_layer(self.step_x)

            self.step_out_h = list()
            for k in range(config['num_stacks']):
                step_h[k] = self.create_cell(str(k), step_pre_processed_x_in, step_h[k])

            self.step_y = O @ step_h[config['num_stacks'] - 1][0]

        # state vars
        self.current_h = self.get_current_h()

        # init the global variables initializer
        tf.set_random_seed(self.config['seed'])
        init = tf.global_variables_initializer()
        self.sess = tf.Session()

        # this represents the writers
        self.train_writer = tf.summary.FileWriter(config['log_dir'] + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(config['log_dir'] + 'test', self.sess.graph)

        # create saver, so training is not discarded
        self.saver = tf.train.Saver()

        # create some summaries
        with tf.name_scope('summaries'):

            # create all summaries
            tf.summary.scalar("mean_squared_error", self.error)
            tf.summary.scalar("absolute_error", self.a_error)
            tf.summary.scalar("learning_rate", self.learning_rate)

        # merge all summaries
        self.summaries = tf.summary.merge_all()
        self.sess.run(init)

    def get_h(self):
        """Gets a reference to the step h."""
        raise NotImplementedError("Get the current h")

    def get_step_h(self):
        """Retrieve the step h"""
        raise NotImplementedError("Retrieve the step hidden state.")

    def get_current_h(self):
        """Deliver current h"""
        raise NotImplementedError("Retrieve the current hidden state.")

    def write_summaries(self, summary, test=False):
        """Depending on the parameter either a test summary or train summary is written."""

        if test:
            self.test_writer.add_summary(summary, self.current_step)

        else:
            self.train_writer.add_summary(summary, self.current_step)

    def save(self):
        """This method savs the model at the specified checkpoint."""
        self.saver.save(self.sess, self.config['log_dir'] + "model.ckpt", self.global_step)

    def init_pre_processing_layer(self):
        """This method basically create a fully connected network using the
        passed structure."""

        with tf.variable_scope("pre_processing", reuse=None):

            # iterate over the structure
            struct_len = len(self.config['pre_process_structure'])
            in_layer = self.config['num_input']

            # iterate over the whole structure and consequently create the
            # corresponding weights
            for k in range(struct_len):

                # create the correct values
                out_layer = self.config['pre_process_structure'][k]
                tf.get_variable("W" + str(k), [out_layer, in_layer], dtype=tf.float32, initializer=self.weights_initializer)
                tf.get_variable("b" + str(k), [out_layer, 1], dtype=tf.float32, initializer=self.bias_initializer)
                in_layer = out_layer

            # create the appropriate weights
            out_layer = self.config['num_hidden']
            tf.get_variable("W" + str(struct_len), [out_layer, in_layer], dtype=tf.float32, initializer=self.weights_initializer)
            tf.get_variable("b" + str(struct_len), [out_layer, 1], dtype=tf.float32, initializer=self.bias_initializer)

    def create_pre_processing_layer(self, x):
        """This method initializes a pre processing network.

        Args:
            x: The input to the pre processing layer."""

        with tf.variable_scope("pre_processing", reuse=True):

            # determine first of all the depth of the network
            struct_len = len(self.config['pre_process_structure'])
            tree = x

            # simply connect all layers
            for k in range(struct_len):
                tree = tf.nn.relu(tf.get_variable("W" + str(k)) @ tree + tf.get_variable("b" + str(k)))

            # pass back the network
            return tf.nn.relu(tf.get_variable("W" + str(struct_len)) @ tree + tf.get_variable("b" + str(struct_len)))

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

            # gradient
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

    def create_cell(self, name, x, h):
        """This method creates a RNN cell. It basically uses the
        previously initialized weights.

        [h, s] = h_state
            new_h: The new hidden vector
        """
        raise NotImplementedError("Please implement create_cell")

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

        # we want to perform n steps
        for k in range(steps):

            self.current_step += 1
            _, summary = self.sess.run([self.minimizer, self.summaries], feed_dict={self.x: trajectories, self.y: target_trajectories})
            self.write_summaries(summary, False)

    def validate(self, trajectories, target_trajectories):
        """This method basically validates the passed trajectories.
        It therefore splits them up so that the future frame get passed as the target.

        Args:
            trajectories: The trajectories to use for validation.
            target_trajectories: The target trajectories

        Returns:
            The error on the passed trajectories
        """

        aerror, summary = self.sess.run([self.a_error, self.summaries], feed_dict={self.x: trajectories, self.y: target_trajectories})
        self.write_summaries(summary, True)
        return aerror

    def init_params(self):
        """This initializes the parameters."""
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def reset(self):
        tf.reset_default_graph()
