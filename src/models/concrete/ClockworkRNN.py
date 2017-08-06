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

from src.models.RecurrentNeuralNetwork import RecurrentNeuralNetwork


class ClockworkRNN(RecurrentNeuralNetwork):
    """This model represents a simple clockwork RNN, based on the
    equally named paper."""

    def __init__(self, config):
        """Constructs a new CW-RNN.

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
                num_modules: The number of modules with different clocks
                module_size: The number of neurons per clocked module
        """

        # create the clockwork mask
        config['unique_name'] = "CWRNN_" + config['unique_name']
        config['num_cells'] = 1
        config['num_hidden'] = config['num_modules'] * config['module_size']

        cw_mask = np.ones((config['num_hidden'], config['num_hidden']), dtype=np.int32)

        # fill the other values with ones
        ms = config['module_size']
        for y in range(1, config['num_modules']):
            for x in range(y):
                cw_mask[ms*y:ms*(y+1), ms*x:ms*(x+1)] = np.zeros((ms, ms), dtype=np.int32)

        # create the constant mask
        self.cw_mask = tf.constant(cw_mask, dtype=tf.float32)

        # create clock periods
        self.clock_periods = np.power(2, np.arange(1, config['num_modules'] + 1) - 1)

        # Perform the super call
        super().__init__(config)

    def get_h(self):
        """Gets a reference to the step h."""
        size = self.config['num_hidden'] * self.config['num_cells']
        h = tf.zeros([size, 1], tf.float32)
        return [h]

    def get_initial_h(self):
        """Gets a reference to the step h."""
        return self.h

    def get_step_h(self):
        """Retrieve the step h"""
        size = self.config['num_hidden'] * self.config['num_cells']
        h = tf.placeholder(tf.float32, [size, 1], name="step_h")
        return [h]

    def get_current_h(self):
        """Deliver current h"""
        size = self.config['num_hidden'] * self.config['num_cells']
        h = np.zeros([size, 1])
        return [h]

    def init_cell(self, name):
        """This method creates the parameters for a cell with
        the given name

        Args:
            name: The name for this cell, e.g. 1
        """
        with tf.variable_scope(name, reuse=None):

            # extract values
            H = self.config['num_hidden']
            I = self.config['num_input']

            # init the layers appropriately
            tf.get_variable("W", [H, I], dtype=tf.float32, initializer=self.weights_initializer)
            tf.get_variable("R", [H, H], dtype=tf.float32, initializer=self.weights_initializer)
            tf.get_variable("b", [H, 1], dtype=tf.float32, initializer=self.bias_initializer)

    def create_cell(self, name, x, h_state, num_cell):
        """This method creates a LSTM cell. It basically uses the
        previously initialized weights.

        Args:
            x: The input to the layer.
            h_state: The hidden input to the layer.

        Returns:
            new_h: The new hidden vector
        """
        [h] = h_state
        out_h = list()

        with tf.variable_scope(name, reuse=True):

            # short form
            ms = self.config['module_size']

            # get matrices appropriately
            WH = tf.get_variable("W")
            RH = self.cw_mask * tf.get_variable("R")
            b = tf.get_variable("b")

            # create the modules itself
            for t in range(self.config['num_modules']):

                # extract the block row
                block_row = tf.slice(RH, [ms * t, 0], [ms, self.config['num_hidden']])

                # make conditional if full or zero
                condition = tf.equal(tf.mod(self.step_num, tf.constant(self.clock_periods[t], dtype=tf.int32)), tf.constant(0))
                filter_row = tf.cond(condition, lambda: tf.identity(block_row), lambda: tf.zeros([ms, self.config['num_hidden']]))

                # retrieve block b and wh
                block_b = tf.slice(b, [ms * t, 0], [ms, 1])
                block_w = tf.slice(WH, [ms * t, 0], [ms, self.config['num_input']])

                # append to output list
                out_h.append(block_w @ x + filter_row @ h + block_b)

            # create new hidden vector
            new_h = tf.concat(out_h, axis=0)

        # pass back the new hidden state
        return [new_h]
