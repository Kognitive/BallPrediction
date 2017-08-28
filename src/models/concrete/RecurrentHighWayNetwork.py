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
        """Constructs a new RHN.

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
        super().__init__(config)

    def get_h(self, stack):
        """Creates a list of tensors representing the hidden state of one cell."""

        return [tf.get_variable("hidden_" + str(stack), [self.config['num_hidden'], 1],
                                dtype=tf.float32,
                                initializer=self.bias_initializer)
                if self.config['learnable_hidden_states']
                else tf.zeros([self.config['num_hidden'], 1], dtype=tf.float32)]

    def init_single_layer(self, name, stack, layer):
        """This method initializes the weights for a single layer, which gets
        used subsequently to specify one gate, e.g. C, H, T.

        Args:
            name: The name of the layer
            stack: The number of the current stack
            layer: The number of the layer
        """

        with tf.variable_scope(name, reuse=None):

            # extract parameters
            H = self.config['num_hidden']

            # Input is only received by first layer
            if layer == 0:
                tf.get_variable("WX", [H, H], dtype=tf.float32, initializer=self.weights_initializer)

            # The hidden state of the previous stack is only available, when there
            # is a next state
            if layer == 0 and stack != 0:
                tf.get_variable("WH", [H, H], dtype=tf.float32, initializer=self.weights_initializer)

            tf.get_variable("R", [H, H], dtype=tf.float32, initializer=self.weights_initializer)
            tf.get_variable("b", [H, 1], dtype=tf.float32, initializer=self.bias_initializer)

    @staticmethod
    def create_single_layer(name, x, h_own, h_prev, activation, stack, layer):
        """This method creates a single layer and returns
        the combined output.

        Args:
            name: The name of the layer
            x: The input from the outside
            h_own: The previous internal hidden state
            h_prev: The hidden state from the the previous stack
            activation: The activation function to use
            stack: The number of the current stack
            layer: The number of the layer
        """

        with tf.variable_scope(name, reuse=True):

            # build up the network
            R = tf.get_variable("R")
            b = tf.get_variable("b")
            term = R @ h_own + b

            if layer == 0 and stack != 0:
                WH = tf.get_variable("WH")
                term = WH @ h_prev + term

            # first layer gets input
            if layer == 0:
                WX = tf.get_variable("WX")
                term = WX @ x + term

            return activation(term)

    def init_rec_highway_layer(self, stack, layer):
        """This method initializes the weights for the specified layer.

        Args:
            stack: The number of the current stack.
            layer: The number of the layer itself.
        """

        with tf.variable_scope(str(layer), reuse=None):

            if not self.config['coupled_gates']:
                self.init_single_layer("C", stack, layer)

            self.init_single_layer("H", stack, layer)
            self.init_single_layer("T", stack, layer)

    def create_rec_highway_layer(self, stack, layer, x, h_own, h_prev):
        """This creates one recursive highway layer, with the previously
        initialized weights.

        Args:
            stack: The number of the current stack
            layer: The number of the layer
            x: The input state
            h_own: The hidden state for this unit.
            h_prev: The hidden state from the previous stacked cell

        Returns:
            The output for this layer
        """

        with tf.variable_scope(str(layer), reuse=True):

            # The input to the layer unit
            H = self.create_single_layer("H", x, h_own, h_prev, self.config['h_node_activation'], stack, layer)
            T = self.create_single_layer("T", x, h_own, h_prev, tf.sigmoid, stack, layer)
            C = tf.constant(1.0) - T if self.config['coupled_gates'] \
                else self.create_single_layer("C", x, h_own, h_prev, tf.sigmoid, stack, layer)

            # create the variables only the first times
            return T * H + C * h_own

    def init_cell(self, stack):
        """This method initializes the variables for the specified stack cell

        Args:
            stack: The number of the stacked cell
        """
        with tf.variable_scope(str(stack), reuse=None):

            # create as much cells as recurrent depth is set to
            for i in range(self.config['recurrence_depth']):

                # init the layers appropriately
                self.init_rec_highway_layer(stack, i)

    def create_cell(self, stack, x, h_own, h_prev):
        """This method creates a RHN cell.

        Args:
            stack: The number of the stacked cell
            x: The input to the layer.
            h_own: The hidden input to the layer
            h_prev: The hidden output of the previous stack cell.

        Returns:
            Effectively a list with one element, where it represents
            the hidden output of this cell.
        """

        with tf.variable_scope(str(stack), reuse=True):

            it_h = h_own[0]
            prev_h = h_prev[0]

            # create as much cells as recurrent depth is set to
            for i in range(self.config['recurrence_depth']):

                # init the layers appropriately
                it_h = self.create_rec_highway_layer(stack, i, x, it_h, prev_h)

        # pass back both states
        return [it_h]
