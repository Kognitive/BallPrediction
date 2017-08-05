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


class LSTM(RecurrentNeuralNetwork):
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
                peephole: True, if peephole connections should be used
        """

        # Perform the super call
        config['unique_name'] = "LSTM_" + config['unique_name']
        super().__init__(config)

    def get_h(self):
        """Gets a reference to the step h."""
        size = self.config['num_hidden'] * self.config['num_cells']
        h = tf.zeros([size, 1], tf.float32)
        s = tf.zeros([size, 1], tf.float32)
        return [h, s]


    def get_initial_h(self):
        """Gets a reference to the step h."""
        return self.h

    def get_step_h(self):
        """Retrieve the step h"""
        size = self.config['num_hidden'] * self.config['num_cells']
        h = tf.placeholder(tf.float32, [size, 1], name="step_h")
        s = tf.placeholder(tf.float32, [size, 1], name="step_s")
        return [h, s]

    def get_current_h(self):
        """Deliver current h"""
        size = self.config['num_hidden'] * self.config['num_cells']
        h = np.zeros([size, 1])
        s = np.zeros([size, 1])
        return [h, s]

    def init_layer(self, name):
        """This method initializes the weights for a layer with the
        given name.

        Args:
            name: The name of the layer.
        """

        with tf.variable_scope(name, reuse=None):

            # short form
            H = self.config['num_hidden']
            I = self.config['num_input']
            C = self.config['num_cells']

            # The input to the layer unit
            tf.get_variable("W", [H, I], dtype=tf.float32, initializer=self.initializer)
            tf.get_variable("R", [H, H * C], dtype=tf.float32, initializer=self.initializer)
            tf.get_variable("b", [H, 1], dtype=tf.float32, initializer=self.initializer)

            # when a peephole is needed
            if self.config['peephole']:
                tf.get_variable("p", [H, 1], dtype=tf.float32, initializer=self.initializer)

    def create_layer(self, name, activation, x, h, s):
        """This method creates one layer, it therefore needs a activation
        function, the name as well as the inputs to the layer.

        Args:
            name: The name of this layer
            activation: The activation to use.
            x: The input state
            h: The hidden state from the previous layer.
            s: The memory state of this layer

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

            # if a peephole should be included
            if self.config['peephole']:
                p = tf.get_variable("p")
                term += tf.multiply(p, s)

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
            self.init_layer("forget_gate")
            self.init_layer("output_gate")
            self.init_layer("input_gate")
            self.init_layer("input_node")

    def create_cell(self, name, x, h_state, num_cell):
        """This method creates a LSTM cell. It basically uses the
        previously initialized weights.

        Args:
            x: The input to the layer.
            h_state: The hidden input to the layer.

        Returns:
            new_h: The new hidden vector
        """
        [h, s] = h_state

        with tf.variable_scope(name, reuse=True):

            sliced_s = tf.slice(s, [self.config['num_hidden'] * num_cell, 0], [self.config['num_hidden'], 1])

            # create all gate layers
            forget_gate = self.create_layer("forget_gate", tf.sigmoid, x, h, sliced_s)
            output_gate = self.create_layer("output_gate", tf.sigmoid, x, h, sliced_s)
            input_gate = self.create_layer("input_gate", tf.sigmoid, x, h, sliced_s)
            input_node = self.create_layer("input_node", tf.tanh, x, h, sliced_s)

            # update input gate
            input_gate = tf.multiply(input_gate, input_node)
            forgotten_memory = tf.multiply(forget_gate, sliced_s)

            # calculate the new s
            new_s = tf.add(input_gate, forgotten_memory)
            new_h = tf.multiply(output_gate, LSTM.lrelu_activation(new_s))

        # pass back both states
        return [new_h, new_s]
