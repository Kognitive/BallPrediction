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


class GatedRecurrentUnit(RecurrentNeuralNetwork):
    """This model represents a GRU recurrent network. It can
    be configured in various ways. This concrete implementation
    features the GRU with forget gates."""

    def __init__(self, unique_name, num_input, num_output, num_hidden, num_cells, num_layers, batch_size, minimizer, seed=3):
        """Constructs a new GRU.

        Args:
            unique_name: Define the unique name of this lstm
            num_input: The number of input units per step.
            num_output: The number of output units per step.
            num_hidden: The number of units in the hidden layer.
            num_cells: The number of cells per layer
            num_layers: Define number of time-step unfolds.
            batch_size: This represents the batch size used for training.
        """

        # Perform the super call
        super().__init__(unique_name, num_input, num_output, num_hidden, num_cells, num_layers, batch_size, minimizer, seed)

    def get_h():
        """Gets a reference to the step h."""
        return [tf.zeros([self.H * self.C, 1], tf.float32)]

    def get_initial_h():
        """Gets a reference to the step h."""
        return self.h

    def get_step_h():
        """Retrieve the step h"""
        raise [tf.placeholder(tf.float32, [self.H * self.C, 1], name="step_h")]

    def get_current_h():
        """Deliver current h"""
        return [np.zeros([self.H * self.C, 1])]

    def init_layer(self, name):
        """This method initializes the weights for a layer with the
        given name.

        Args:
            name: The name of the layer.
        """

        with tf.variable_scope(name, reuse=None):

            # The input to the layer unit
            tf.get_variable("W", [self.H, self.I], dtype=tf.float32, initializer=self.initializer)
            tf.get_variable("R", [self.H, self.H * self.C], dtype=tf.float32, initializer=self.initializer)
            tf.get_variable("b", [self.H, 1], dtype=tf.float32, initializer=self.initializer)

    def create_layer(self, name, activation, x, h):
        """This method creates one layer, it therefore needs a activation
        function, the name as well as the inputs to the layer.

        Args:
            name: The name of this layer
            activation: The activation to use.
            x: The input state
            h: The hidden state from the previous layer.

        Returns:
            The output for this layer
        """

        with tf.variable_scope(name, reuse=True):

            # The input to the layer unit
            W = tf.get_variable("W")
            R = tf.get_variable("R")
            b = tf.get_variable("b")

            # create the term
            term = W @ x + R @ h + b

            # create the variables only the first times
            return activation(term)

    def init_cell(self, name):
        """This method creates the parameters for a cell with
        the given name

        Args:
            name: The name for this cell, e.g. 1
        """
        with tf.variable_scope(name, reuse=None):

            # init the layers appropriately
            self.init_layer("recurrent_gate")
            self.init_layer("input_gate")
            self.init_layer("input_node")

    def create_cell(self, name, x, h_state):
        """This method creates a GRU cell. It basically uses the
        previously initialized weights.

        Args:
            x: The input to the layer.
            h: The hidden input to the layer.

        Returns:
            new_h: The new hidden vector
        """

        [h] = h_state

        with tf.variable_scope(name, reuse=True):

            # create all gate layers
            recurrent_gate = self.create_layer("recurrent_gate", tf.sigmoid, x, h)
            mod_h = tf.multiply(recurrent_gate, h)

            input_gate = self.create_layer("input_gate", tf.sigmoid, x, h)
            input_node = self.create_layer("input_node", tf.tanh, x, mod_h)

            # update input gate
            right_input_node = tf.multiply(input_gate, input_node)
            left_input_node  = tf.multiply(tf.ones([]) - input_gate, h)

            # calculate the new s
            new_h = tf.add(left_input_node, right_input_node)

        # pass back both states
        return [new_h]
