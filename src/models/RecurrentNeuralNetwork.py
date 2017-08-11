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
from src.models.HighwayNetwork import HighwayNetwork
from src.models.concrete.RecurrentHighWayNetworkCell import RecurrentHighWayNetworkCell


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

        with tf.variable_scope(config['unique_name']):

            # init some variables
            self.pre_highway_network = self.get_preprocess_network()

            # initialize all cells
            self.cells = self.__init_all_cells()

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
            self.h = [cell.get_hidden_state() for cell in self.cells]

            # Unstack the input itself
            unstacked_x = tf.unstack(self.x, axis=1)

            # Combine all 3 networks to one
            processed_unstacked_x = self.get_input_to_hidden_network(unstacked_x)
            h = self.get_hidden_to_hidden_network(config, processed_unstacked_x)
            self.target_y = self.get_hidden_to_output_network(h)

            # first of create the reduced squared error
            red_squared_err = tf.reduce_sum(tf.pow(self.target_y - self.y, 2), axis=0)

            # So far we have got the model
            self.error = 0.5 * tf.reduce_mean(red_squared_err, axis=0)
            self.a_error = tf.reduce_mean(tf.sqrt(red_squared_err), axis=0)
            self.minimizer = self.create_minimizer(self.learning_rate, self.error, self.global_step)

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

    def __init_all_cells(self):

        # create the common
        cell_config = {}
        cell_config['num_layers'] = self.config['recurrence_depth']
        cell_config['coupled_gates'] = self.config['coupled_gates']
        cell_config['learn_hidden'] = self.config['learnable_hidden_states']
        cell_config['h_activation'] = self.config['h_node_activation']
        cell_config['num_input'] = self.config['num_input']
        cell_config['num_hidden'] = self.config['num_hidden']
        cell_config['seed'] = self.config['seed']

        # init combined cells
        cells = list()
        for k in range(self.config['num_stacks']):

            cell_config['cell_name'] = str(k)
            cell_config['head_of_stack'] = k == 0
            cells.append(RecurrentHighWayNetworkCell(cell_config))

        return cells

    def get_hidden_to_hidden_network(self, config, processed_unstacked_x):

        # use for dynamic
        h = self.h

        # unfold the cell
        for x_in in processed_unstacked_x:

            # create a cell
            h[0] = self.cells[0].create_cell(x_in, h[0], [None])

            # stack them up
            for k in range(1, config['num_stacks']):
                h[k] = self.cells[0].create_cell(x_in, h[k], h[k - 1])

        return h

    def get_input_to_hidden_network(self, unstacked_x):
        """This method should take a list of input variables and
        process them through the same highway network.

        Args:
            unstacked_x: The list of input variables.

        Returns
            A list of processed variables.

        """
        return [self.pre_highway_network.get_graph(x_in) for x_in in unstacked_x]

    def get_hidden_to_output_network(self, h):
        """This method creates the hidden to output layer.

        Args:
            h: A list of list, containing the hidden states for each layer.

        Returns:
            The output layer itself.
        """

        # Create the hidden to output layer
        with tf.variable_scope("hidden_to_output"):

            # Initialize the weights used for the output layer
            pre_target_h_weights = list()
            pre_target_h_bias = tf.get_variable("b", [self.config['num_output'], 1],
                                                dtype=tf.float32,
                                                initializer=self.bias_initializer)

            for k in range(self.config['num_stacks']):
                pre_target_h_weights.append(tf.get_variable("W" + str(k),
                                                            [self.config['num_output'], self.config['num_hidden']],
                                                            dtype=tf.float32,
                                                            initializer=self.weights_initializer))

            # For an easier propagation of the gradient, we want to connect the
            # outputs of all hidden units to the final output. To do so we multiply
            # the previously created matrices with each hidden vector
            pre_target_y = pre_target_h_weights[0] @ h[0][0]
            for k in range(1, self.config['num_stacks']):

                # sum them all up using the weight matrices
                pre_target_y += pre_target_h_weights[k] @ h[k][0]

            # finally combine them to the target vector
            return self.config['activation_output_layer'](pre_target_y + pre_target_h_bias)

    def get_preprocess_network(self):
        """This method delivers the pre processing network."""

        # create the configuration for the pre processing network
        highway_conf = {
            'num_input': self.config['num_input'],
            'num_intermediate': self.config['num_intermediate'],
            'num_output': self.config['num_hidden'],
            'num_preprocess_layers': self.config['num_preprocess_layers'],
            'preprocess_coupled_gates': self.config['preprocess_coupled_gates'],
            'preprocess_activation': self.config['preprocess_activation'],
            'preprocess_h_node_activation': self.config['preprocess_h_node_activation'],
            'seed': self.config['seed']
        }

        return HighwayNetwork(highway_conf)

    def write_summaries(self, summary, test=False):
        """Depending on the parameter either a test summary or train summary is written."""

        if test:
            self.test_writer.add_summary(summary, self.current_step)

        else:
            self.train_writer.add_summary(summary, self.current_step)

    def save(self):
        """This method savs the model at the specified checkpoint."""
        self.saver.save(self.sess, self.config['log_dir'] + "model.ckpt", self.global_step)

    def create_minimizer(self, learning_rate, error, global_step):
        """This method creates the correct optimizer."""

        # first of all set the global step to zero, because we train from the beginning
        minimizer = 0

        # choose between the available optimizers
        str_minimizer = self.config['minimizer']
        if str_minimizer == 'momentum':
            minimizer = tf.train.MomentumOptimizer(learning_rate, self.config['momentum'], use_nesterov=True)

        elif str_minimizer == 'adam':
            minimizer = tf.train.AdamOptimizer()

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
