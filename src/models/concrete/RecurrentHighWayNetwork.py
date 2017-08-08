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
        return [tf.zeros([self.config['num_hidden'], 1], tf.float32)]

    def get_step_h(self):
        """Retrieve the step h"""
        return [tf.placeholder(tf.float32, [self.config['num_hidden'], 1], name="step_h")]

    def get_current_h(self):
        """Deliver current h"""
        return [np.zeros([self.config['num_hidden'], 1])]

    def init_rec_highway_layer(self, layer):
        """This method initializes the weights for a layer with the
        given name.

        Args:
            layer: The number of the layer itself..
            first: Specifies whether this is the first highway layer
        """

        with tf.variable_scope(str(layer), reuse=None):

            if not self.config['coupled_gates']:
                self.init_single_layer("C", layer)

            self.init_single_layer("H", layer)
            self.init_single_layer("T", layer)

    def init_single_layer(self, name, layer):
        """This method creates a single layer.

        Args:
            name: The name of the layer
            layer: The number of the layer
        """
        with tf.variable_scope(name, reuse=None):

            # extract parameters
            H = self.config['num_hidden']

            if layer == 0:
                tf.get_variable("W", [H, H], dtype=tf.float32, initializer=self.weights_initializer)

            tf.get_variable("R", [H, H], dtype=tf.float32, initializer=self.weights_initializer)
            tf.get_variable("b", [H, 1], dtype=tf.float32, initializer=self.bias_initializer)

    def create_rec_highway_layer(self, layer, x, h):
        """This method creates one layer, it therefore needs a activation
        function, the name as well as the inputs to the layer.

        Args:
            layer: The number of the layer
            x: The input state
            h: The hidden state from the previous layer.

        Returns:
            The output for this layer
        """

        with tf.variable_scope(str(layer), reuse=True):

            # The input to the layer unit
            H = self.create_single_layer("H", x, h, self.config['h_node_activation'], layer)
            T = self.create_single_layer("T", x, h, tf.sigmoid, layer)
            C = tf.constant(1.0) - T if self.config['coupled_gates'] else self.create_single_layer("C", x, h, tf.sigmoid, layer)

            # create the variables only the first times
            return T * H + C * h

    @staticmethod
    def create_single_layer(name, x, h, activation, layer):
        """This method creates a single layer and returns
        the combined output.

        Args:
            name: The name of the layer
            x: The input from the outside
            h: The previous hidden state
            activation: The activation function to use
            layer: The number of the layer
        """
        with tf.variable_scope(name, reuse=True):

            # build up the network
            R = tf.get_variable("R")
            b = tf.get_variable("b")
            term = R @ h + b

            # first layer gets input
            if layer == 0:
                W = tf.get_variable("W")
                term = W @ x + term

            return activation(term)

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
                self.init_rec_highway_layer(i)

    def create_cell(self, name, x, h):
        """This method creates a LSTM cell. It basically uses the
        previously initialized weights.

        Args:
            name: The name for this cell, e.g. 1
            x: The input to the layer.
            h: The hidden input to the layer.

        Returns:
            new_h: The new hidden vector
        """

        with tf.variable_scope(name, reuse=True):

            it_h = h[0]

            # create as much cells as recurrent depth is set to
            for i in range(self.config['recurrence_depth']):

                # init the layers appropriately
                it_h = self.create_rec_highway_layer(i, x, it_h)

        # pass back both states
        return [it_h]
