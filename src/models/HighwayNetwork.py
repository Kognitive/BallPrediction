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

import tensorflow as tf


class HighwayNetwork:
    """This class represents a processing highway network."""

    def __init__(self, config):
        """Constructor for a highway network, which can be used for pre processing

        Args:
            config:
                num_input: The number of the inputs
                num_intermediate: The number of neurons for each highway layer
                num_output: The number of the output layer for this pre processing network.
                num_preprocess_layers: The number of highway layers
                preprocess_coupled_gates: True, if the gates should be coupled
                preprocess_activation: The activation for the FC layers
                preprocess_h_node_activation: The activation for the preprocess h node
        """

        # save hyper parameters
        self.weights_initializer = tf.contrib.layers.variance_scaling_initializer(1.0, 'FAN_AVG', True, config['seed'])
        self.bias_initializer = tf.constant_initializer(0.0)

        # save config
        self.config = config
        with tf.variable_scope("preprocess_network", reuse=None):

            I = self.config['num_input']
            H = self.config['num_intermediate']
            O = self.config['num_output']

            tf.get_variable("W_in", [H, I], dtype=tf.float32, initializer=self.weights_initializer)
            tf.get_variable("b_in", [H, 1], dtype=tf.float32, initializer=self.bias_initializer)

            # init the high way layers as well
            for k in range(self.config['num_preprocess_layers']):
                self.init_highway_layer(k)

            tf.get_variable("W_out", [O, H], dtype=tf.float32, initializer=self.weights_initializer)
            tf.get_variable("b_out", [O, 1], dtype=tf.float32, initializer=self.bias_initializer)

    def get_graph(self, x):
        """Simply get the graph of this highway network."""

        with tf.variable_scope("preprocess_network", reuse=True):
            tree = self.get_activation(self.config['preprocess_activation'])(tf.get_variable("W_in") @ x + tf.get_variable("b_in"))

            # init the high way layers as well
            for k in range(self.config['num_preprocess_layers']):
                tree = self.create_highway_layer(k, tree)

            return self.get_activation(self.config['preprocess_activation'])(tf.get_variable("W_out") @ tree + tf.get_variable("b_out"))

    def init_highway_layer(self, layer):
        """This method initializes the weights for a layer with the
        given name.

        Args:
            layer: The number of the layer itself..
        """

        with tf.variable_scope(str(layer), reuse=None):
            if not self.config['preprocess_coupled_gates']:
                self.init_single_layer("C")

            self.init_single_layer("H")
            self.init_single_layer("T")

    def init_single_layer(self, name):
        """This method creates a single layer.

        Args:
            name: The name of the layer
        """
        with tf.variable_scope(name, reuse=None):

            # extract parameters
            H = self.config['num_intermediate']

            tf.get_variable("W", [H, H], dtype=tf.float32, initializer=self.weights_initializer)
            bias_initializer = tf.constant_initializer(-1.0) if name == "T" else self.bias_initializer
            tf.get_variable("b", [H, 1], dtype=tf.float32, initializer=bias_initializer)

    def create_highway_layer(self, layer, x):
        """This method creates one layer, it therefore needs a activation
        function, the name as well as the inputs to the layer.

        Args:
            layer: The number of the layer
            x: The input state

        Returns:
            The output for this layer
        """

        with tf.variable_scope(str(layer), reuse=True):
            # The input to the layer unit
            H = self.create_single_layer("H", x, self.get_activation(self.config['preprocess_h_node_activation']))
            T = self.create_single_layer("T", x, self.get_activation('sigmoid'))
            C = tf.constant(1.0) - T if self.config['preprocess_coupled_gates'] else self.create_single_layer("C", x, self.get_activation('sigmoid'))

            # create the variables only the first times
            return tf.nn.dropout(T * H + C * x, self.config['dropout_prob'])

    def get_activation(self, name):
        if name == 'tanh':
            return tf.nn.tanh
        elif name == 'sigmoid':
            return tf.nn.sigmoid
        elif name == 'identity':
            return tf.identity
        elif name == 'lrelu':
            return lambda x: tf.maximum(x, 0.01 * x)

    @ staticmethod
    def create_single_layer(name, x, activation):
        """This method creates a single layer and returns
        the combined output.

        Args:
            name: The name of the layer
            x: The input from the outside
            activation: The activation function to use
        """
        with tf.variable_scope(name, reuse=True):

            # build up the network
            W = tf.get_variable("W")
            b = tf.get_variable("b")
            term = W @ x + b

            return activation(term)