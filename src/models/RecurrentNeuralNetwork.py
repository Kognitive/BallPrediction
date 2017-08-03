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

    def __init__(self, unique_name, num_input, num_output, num_hidden, num_cells, num_layers, batch_size, minimizer, seed=3):
        """Constructs a new RNN.

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
        super().__init__("RNN_" + unique_name, 1)

        # save hyper parameters
        self.I = num_input
        self.O = num_output
        self.H = num_hidden
        self.C = num_cells
        self.N = num_layers
        self.BS = batch_size
        self.initializer = tf.random_normal_initializer(0.0, 0.1, seed=seed)

        with tf.variable_scope(unique_name):

            # init
            self.init_common()

            # initialize num_cells different cells
            for i in range(num_cells):
                self.init_cell(str(i))

            # --------------------------- TRAINING ----------------------------

            # create a tensor for the input
            self.x = tf.placeholder(tf.float32, [self.I, self.N, None], name="input")
            self.y = tf.placeholder(tf.float32, [self.O, self.N, None], name="target")
            self.learning_rate = tf.placeholder(tf.float32, [], name="learning_rate")

            # define the memory state
            self.h = self.get_h()

            # unstack the input
            unstacked_x = tf.unstack(self.x, axis=1)

            # use for dynamic
            h = self.get_initial_h_node()

            # create weights
            outputs = list()

            # unfold the cell
            for x_in in unstacked_x:

                # create a cell
                y, h = self.create_combined_cell(x_in, h)
                outputs.append(y)

            # the final states
            self.target_y = tf.stack(outputs, axis=1)

            # So far we have got the model
            self.error = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.pow(self.target_y - self.y, 2), axis=0))
            self.minimizer = minimizer(self.learning_rate).minimize(self.error)

            # ------------------------------ EVALUATION ---------------------------------

            # here comes the step model
            self.step_x = tf.placeholder(tf.float32, [self.I, 1], name="step_x")
            self.step_h = self.get_step_h()

            # the model
            self.step_y, self.step_out_h = self.create_combined_cell(self.step_x, self.step_h)

        # state vars
        self.current_h = self.get_current_h()

        # init the global variables initializer
        tf.set_random_seed(4)
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def get_h():
        """Gets a reference to the step h."""
        raise NotImplementedError("Get the current h")

    def get_initial_h():
        """Gets a reference to the step h."""
        raise NotImplementedError("Get the current initial h")

    def get_step_h():
        """Retrieve the step h"""
        raise NotImplementedError("Retrieve the step hidden state.")

    def get_current_h():
        """Deliver current h"""
        raise NotImplementedError("Retrieve the current hidden state.")

    def init_common(self):
        """This initializes the common variables.
        """
        with tf.variable_scope("common", reuse=None, ):
            tf.get_variable("O", [self.O, self.H * self.C], dtype=tf.float32,
                            initializer=self.initializer)

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

    def create_combined_cell(self, x, h):
        """This method creates a combined rnn cell. It basically
        connects them appropriately.

        Args:
            x: The input vector to the cell (self.I)
            h: The hidden vector (self.I * self.num_cells)

        Returns:
            Tuple(y, h, s), where y is the predicted output,
            h is the hidden vector for all cells
        """

        # create all cells
        all_new_h = list()
        for k in range(self.C):
            new_h = self.create_lstm_cell(str(k), x, h)
            all_new_h.append(new_h)

        # stack the hidden vectors
        all_new_h = tf.concat(all_new_h, 0)

        # create the output
        with tf.variable_scope("common", reuse=True):
            O = tf.get_variable("O")
            y = O @ all_new_h

        # pass back the cell outputs
        return y, all_new_h

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
raise NotImplementedError("Get the current h")
    def train(self, trajectories, steps, learning_rate):
        """This method retrieves a list of trajectories. It can further
        process or transform these trajectories. But the desired overall
        behaviour is, that it should use the passed trajectories as training
        data and thus adjust the parameters of the model appropriately.

        Args:
            trajectories: This is a list of trajectories (A trajectory is a numpy vector)
            steps: The number of steps the model should execute.
            learning_rate: The learning rate for this step
        """

        # sample them randomly according to the batch size
        slices = np.random.randint(0, np.size(trajectories, 2), self.BS)
        traj = trajectories[:, :-1, slices]
        target = trajectories[:, 1:, slices]

        # we want to perform n steps
        for k in range(steps):
            self.sess.run(self.minimizer, feed_dict={self.x: traj, self.y: target, self.learning_rate: learning_rate})

    def validate(self, trajectories):
        """This method basically validates the passed trajectories.
        It therefore splits them up so that the future frame get passed as the target.

        Args:
            trajectories: The trajectories to use for validation.

        Returns:
            The error on the passed trajectories
        """
        traj = trajectories[:, :-1, :]
        target = trajectories[:, 1:, :]

        return self.sess.run(self.error, feed_dict={self.x: traj, self.y: target})

    def init_params(self):
        """This initializes the parameters."""
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def reset(self):
        tf.reset_default_graph()
