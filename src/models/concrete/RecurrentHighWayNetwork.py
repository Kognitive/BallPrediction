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


class RecurrentHighWayNetwork(RecurrentNeuralNetwork):
    """This model represents a LSTM recurrent network. It can
    be configured in various ways. This concrete implementation
    features the LSTM with forget gates."""

    def __init__(self, config):
        """Constructs a new LSTM.

        Args:
            config: The configuration parameters
                unique_name: Define the unique name of this lstm
                num_input: The number of input units per step.
                num_output: The number of output units per step.
                num_hidden: The number of units in the hidden layer.
                num_cells: The number of cells per layer
                num_layers: Define number of time-step unfolds.
                batch_size: This represents the batch size used for training.
                minimizer: Select the appropriate minimizer
                seed: Represents the seed for this model
                momentum: The momentum if the minimizer is momentum
                lr_rate: The initial learning rate
                lr_decay_steps: The steps until a decay should happen
                lr_decay_rate: How much should the learning rate be reduced
                recurrence_depth: The recurrence depth per cell.
        """

        # Perform the super call
        config['unique_name'] = "RHN_" + config['unique_name']
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

    def init_highway_layer(self, name, first):
        """This method initializes the weights for a layer with the
        given name.

        Args:
            name: The name of the layer.
            first: Specifies whether this is the first highway layer
        """

        with tf.variable_scope(name, reuse=None):

            self.init_single_layer("C", first)
            self.init_single_layer("H", first)
            self.init_single_layer("T", first)

    def init_single_layer(self, name, first):
        """This method creates a single layer.

        Args:
            name The sub name of the layer inside of the RHN
        """
        with tf.variable_scope(name, reuse=None):

            # extract parameters
            H = self.config['num_hidden']
            C = self.config['num_cells']
            I = self.config['num_input']

            hidden_size = H * (C if first else 1)

            if first:
                tf.get_variable("W", [H, I], dtype=tf.float32, initializer=self.initializer)

            tf.get_variable("R", [H, hidden_size], dtype=tf.float32, initializer=self.initializer)
            tf.get_variable("b", [H, 1], dtype=tf.float32, initializer=self.initializer)

    def create_highway_layer(self, name, x, h_state, num_cell):
        """This method creates one layer, it therefore needs a activation
        function, the name as well as the inputs to the layer.

        Args:
            name: The name of this layer
            x: The input state
            h_state: The hidden state from the previous layer.
            num_cell: The number of the cell.

        Returns:
            The output for this layer
        """
        h = h_state
        first = num_cell == 0

        NH = self.config['num_hidden']
        slice_h = h if not first else tf.slice(h, [NH * num_cell, 0], [NH, 1])

        with tf.variable_scope(name, reuse=True):

            # The input to the layer unit
            C = self.create_single_layer("C", x, h, tf.sigmoid, first)
            H = self.create_single_layer("H", x, h, tf.tanh, first)
            T = self.create_single_layer("T", x, h, tf.sigmoid, first)

            # create the variables only the first times
            return T * H + C * slice_h

    @staticmethod
    def create_single_layer(name, x, h, activation, first):
        """This method creates a single layer and returns
        the combined output.

        Args:
            name: The name of the layer
            x: The input from the outside
            h: The previous hidden state
            activation: The activation function to use
            first: True, if it is the first in a cell.
        """
        with tf.variable_scope(name, reuse=True):

            fp = tf.get_variable("W") @ x if first else 0
            R = tf.get_variable("R")
            b = tf.get_variable("b")
            return activation(fp + R @ h + b)

    def init_cell(self, name):
        """This method creates the parameters for a cell with
        the given name

        Args:
            name: The name for this cell, e.g. 1
        """
        with tf.variable_scope(name, reuse=None):

            # create as much cells as recurrent depth is set to
            for i in range(self.config['recurrence_depth']):

                # init the layers appropriately
                self.init_highway_layer(str(i), i == 0)

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

        with tf.variable_scope(name, reuse=True):

            it_h = h

            # create as much cells as recurrent depth is set to
            for i in range(self.config['recurrence_depth']):
                # init the layers appropriately
                it_h = self.create_highway_layer(str(i), x, it_h, i)

        # pass back both states
        return [it_h]
