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


class RecurrentHighWayNetworkCell:
    """This class represents a recurrent highway network cell."""

    def __init__(self, config):
        """Constructs a new cell from which you can obtain arbitrary graphs.

        Args:
            config: The configuration parameters
                cell_name: The name of this cell
                num_layers: The number of recurrent layers inside of this cell
                coupled_gates: Boolean, stating whether C = 1 - T (Default = True)
                head_of_stack: True, if you don't want to input the hidden state of a different cell (Default = True)
                learn_hidden: True, if you want the hidden state to be a variable which can be learner (Default = False)

                h_activation: The activation function of the H gate
                t_activation: The activation function of the T gate (Default = tf.sigmoid)
                c_activation: The activation function of the C gate (Default = tf.sigmoid)

                layer_normalization: True, if you want layer normalization to be activated.

                num_input: The number of input units
                num_hidden: The number of hidden units

                seed: Represents the seed for this model
        """

        # list of the necessary arguments
        required_arguments = ['cell_name', 'num_layers', 'num_input', 'num_hidden', 'h_activation']
        for arg in required_arguments:
            if arg not in config:
                raise AttributeError("You have to supply the attribute {}".format(arg))

        # check whether to apply the default settings
        if 'head_stack' not in config: config['head_stack'] = True
        if 'coupled_gates' not in config: config['coupled_gates'] = True
        if 'learn_hidden' not in config: config['learn_hidden'] = False
        if 't_activation' not in config: config['t_activation'] = tf.sigmoid
        if 'c_activation' not in config: config['c_activation'] = tf.sigmoid
        if 'layer_normalization' not in config: config['layer_normalization'] = False
        if 'seed' not in config: config['seed'] = None

        # save the configuration for internal usage
        self.config = config

        # create the initializer for the various layers. Use xavier initialization for the normal
        # weights and use a zero initializer for the bias of the standard gate. The T gate differs
        # from that, because we want to bias the network at startup to let the input through
        self.weights_initializer = tf.contrib.layers.variance_scaling_initializer(1.0, 'FAN_AVG', True, config['seed'])
        self.bias_initializer = tf.constant_initializer(0.0)
        self.t_gate_bias_initializer = tf.constant_initializer(-1.0)

        # introduce the variable scope first
        with tf.variable_scope(config['cell_name']):
            self.__init_structure()

    def __init_structure(self):
        """This method initializes the variables for the specified stack cell"""

        self.hidden_state = self.__init_hidden_state(self.bias_initializer)

        # create as much cells as recurrent depth is set to
        for i in range(self.config['num_layers']):

            # init the layers appropriately
            self.__init_highway_layer(i)

    def __init_highway_layer(self, layer):
        """This method initializes the weights for the specified layer.

        Args:
            layer: The number of the layer relative to the cell.
        """

        with tf.variable_scope("layer_{}".format(layer), reuse=None):
            if not self.config['coupled_gates']:
                self.__init_single_layer("C", layer, self.weights_initializer, self.bias_initializer)

            self.__init_single_layer("H", layer, self.weights_initializer, self.bias_initializer)
            self.__init_single_layer("T", layer, self.weights_initializer, self.t_gate_bias_initializer)

    def __init_single_layer(self, name, layer, weight_init, bias_init):
        """This method initializes the weights for a single layer, which gets
        used subsequently to specify one gate inside of one layer, e.g. C, H, T.

        Args:
            name: The name of the layer, e.g. C, H, T
            layer: The number of the layer
            weight_init: The initializer of the weights
            bias_init: The initializer of the bias
        """

        with tf.variable_scope(name, reuse=None):

            # extract parameters
            num_hidden = self.config['num_hidden']

            # Input is only received by first layer
            if layer == 0:
                tf.get_variable("WX", [num_hidden, num_hidden], dtype=tf.float32, initializer=weight_init)

            # The hidden state of the previous stack is only available, when there
            # is a next state
            if layer == 0 and not self.config['head_of_stack']:
                tf.get_variable("WH", [num_hidden, num_hidden], dtype=tf.float32, initializer=weight_init)

            # check whether we have to add layer normalization
            if self.config['layer_normalization']:
                tf.get_variable("g", [num_hidden, 1], dtype=tf.float32, initializer=weight_init)

            tf.get_variable("R", [num_hidden, num_hidden], dtype=tf.float32, initializer=weight_init)
            tf.get_variable("b", [num_hidden, 1], dtype=tf.float32, initializer=bias_init)

    def __init_hidden_state(self, init):
        """Initialize the hidden state of this cell."""

        return [tf.get_variable("hidden_state", [self.config['num_hidden'], 1],
                               dtype=tf.float32,
                               trainable=self.config['learn_hidden'],
                               initializer=init)]

    def get_hidden_state(self):
        return self.hidden_state

    def __create_single_layer(self, name, x, h_own, h_prev, activation, layer):
        """This method creates a single layer and returns
        the combined output.

        Args:
            name: The name of the layer
            x: The input from the outside
            h_own: The previous internal hidden state
            h_prev: The hidden state from the the previous stack
            activation: The activation function to use
            layer: The number of the layer
        """

        with tf.variable_scope(name, reuse=True):

            # build up the network
            R = tf.get_variable("R")
            term = R @ h_own

            if layer == 0 and not self.config['head_stack']:
                WH = tf.get_variable("WH")
                term = WH @ h_prev + term

            # first layer gets input
            if layer == 0:
                WX = tf.get_variable("WX")
                term = WX @ x + term

            if self.config['layer_normalization']:
                mean = tf.reduce_sum(term, axis=0) / self.config['num_hidden']
                variance = tf.sqrt(tf.reduce_sum(tf.pow(term - mean, 2)) / self.config['num_hidden'])

                g = tf.get_variable('g')
                term = (g / variance) * (term - mean)

            # append the bias
            b = tf.get_variable("b")
            term += b

            return activation(term)

    def __create_highway_layer(self, layer, x, h_own, h_prev):
        """This creates one recursive highway layer, with the previously
        initialized weights.

        Args:
            layer: The number of the layer
            x: The input state
            h_own: The hidden state for this unit.
            h_prev: The hidden state from the previous stacked cell

        Returns:
            The output for this layer
        """

        with tf.variable_scope("layer_{}".format(layer), reuse=True):

            # The input to the layer unit
            H = self.__create_single_layer("H", x, h_own, h_prev, self.config['h_activation'], layer)
            T = self.__create_single_layer("T", x, h_own, h_prev, self.config['t_activation'], layer)
            C = tf.constant(1.0) - T if self.config['coupled_gates'] \
                else self.__create_single_layer("C", x, h_own, h_prev, self.config['c_activation'], layer)

            # create the variables only the first times
            return T * H + C * h_own

    def create_cell(self, x, h_own, h_prev):
        """This method creates a RHN cell.

        Args:
            x: The input to the layer.
            h_own: The hidden input to the layer
            h_prev: The hidden output of the previous stack cell.

        Returns:
            Effectively a list with one element, where it represents
            the hidden output of this cell.
        """

        with tf.variable_scope(self.config['cell_name'], reuse=True):

            [it_h] = h_own

            # create as much cells as recurrent depth is set to
            for i in range(self.config['num_layers']):

                # init the layers appropriately
                it_h = self.__create_highway_layer(i, x, it_h, h_prev)

        # pass back both states
        return [it_h]
