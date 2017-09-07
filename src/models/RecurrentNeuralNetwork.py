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

        # Save configuration and call the base class
        self.config = config
        super().__init__(config['unique_name'])

        # use the name of the model as a first variable scope
        with tf.variable_scope(config['unique_name']):

            # ------------------------ INITIALIZATION ----------------------------

            # create initializers and use xavier initialization for the weights
            # and use a bias of zero
            self.bias_initializer = tf.constant_initializer(0.0)
            self.weights_initializer =\
                tf.contrib.layers.variance_scaling_initializer(1.0, 'FAN_AVG', True, config['seed'])

            # just a placeholder indicating whether this is training time or not
            self.training_time = tf.placeholder(tf.bool, None, name="training_time")

            # create preprocess and postprocess network. Both will be
            # modeled as a highway network
            self.pre_highway_network = self.get_preprocess_network()
            self.post_highway_network = self.get_postprocess_network()

            # initialize all cells
            self.cells = self.__init_all_cells()

            # ----------------------- VARIABLES & PLACEHOLDER ---------------------------

            self.global_step = tf.Variable(0, trainable=False, name='global_step')
            self.global_episode = tf.Variable(0, trainable=False, name='global_episode')

            # X and Y Tensor
            self.x = tf.placeholder(tf.float32, [config['num_input'],
                                                 config['rec_num_layers'] + config['rec_num_layers_teacher_forcing'],
                                                 None], name="input")

            self.y = tf.placeholder(tf.float32, [config['num_output'],
                                                 config['rec_num_layers_student_forcing'] + config['rec_num_layers_teacher_forcing'] + 1,
                                                 None], name="target")

            if config['num_class_slots'] > 0:
                self.labels = tf.placeholder(tf.int32, [config['rec_num_layers_student_forcing'] + config[
                                                         'rec_num_layers_teacher_forcing'] + 1, None], name="labels")
                lst_target_labels = tf.unstack(self.labels, axis=0)

            else:
                lst_target_labels = None

            # --------------------------------- GRAPH ------------------------------------

            # define the memory state
            self.h = [tf.tile(cell.get_hidden_state(), [1, tf.shape(self.x)[2]]) for cell in self.cells]

            normalized_x = self.x + tf.random_normal(tf.shape(self.x), 0.0, 0.01)
            to_use_x = tf.cond(self.training_time, lambda: tf.identity(normalized_x), lambda: tf.identity(self.x)) \
                if config['add_variance'] else self.x

            # unstack the input to a list, so it can be easier processed
            unstacked_x = tf.unstack(to_use_x, axis=1)

            # create all 3 components of the network, from preprocess, recurrent and
            # postprocess parts of the network.
            processed_unstacked_x = self.get_input_to_hidden_network(unstacked_x)
            lst_h = self.get_hidden_to_hidden_network(config, processed_unstacked_x, self.h)
            cut_lst_h = lst_h[-(config['rec_num_layers_teacher_forcing'] + 1):]

            # create the outputs for each element in the list
            lst_output, pred_labels, class_error, loss = self.get_hidden_to_output_network(cut_lst_h, lst_target_labels)

            # apply some student forcing
            for self_l in range(config['rec_num_layers_student_forcing']):
                added_model = lst_output[-1] + (0 if not config['distance_model'] else unstacked_x[-1])
                unstacked_x.append(added_model)
                processed_self_x_in = self.get_input_to_hidden_network([added_model])
                h = self.get_hidden_to_hidden_network(config, processed_self_x_in, cut_lst_h[-1])
                outputs, _, _, _ = self.get_hidden_to_output_network(h, lst_target_labels)
                lst_output.append(outputs[0])

            # define the target y
            if len(pred_labels) > 0:
                self.target_labels = tf.stack(pred_labels, axis=1, name="target_labels")
            self.target_y = tf.stack(lst_output, axis=1, name="target_y")
            if config['distance_model']:
                self.target_y = tf.cumsum(self.target_y, axis=1) + tf.expand_dims(unstacked_x[-1], axis=1)

            # first of create the reduced squared error
            err = self.target_y - self.y
            squared_err = tf.pow(err, 2)

            self.error = 0.5 * tf.reduce_mean(squared_err) + loss
            self.a_error = tf.reduce_mean(tf.sqrt(tf.reduce_sum(squared_err, axis=0)), name="a_error")
            self.single_error = tf.reduce_mean(tf.reduce_mean(tf.abs(err), axis=1), axis=1, name="single_error")

            # determine classification rate
            self.classification_rate = class_error / tf.cast(tf.size(self.labels), tf.float32)

            # increment global episode
            self.inc_global_episode = tf.assign(self.global_episode,
                                                self.global_episode + 1,
                                                name='inc_global_episode')

            # create minimizer
            self.learning_rate = tf.train.exponential_decay(
                self.config['lr_rate'],
                self.global_step,
                self.config['lr_decay_steps'],
                self.config['lr_decay_rate'],
                staircase=False)

            # create the minimizer
            self.minimizer = self.create_minimizer(self.learning_rate, self.error, self.global_step)

        # init the global variables initializer
        tf.set_random_seed(self.config['seed'])

        # create a new session for running the model
        # the session is not visible to the callee
        self.sess = tf.Session()

        # this represents the writers
        self.train_writer = tf.summary.FileWriter(config['log_dir'] + 'train', self.sess.graph)
        self.test_writer = tf.summary.FileWriter(config['log_dir'] + 'test', self.sess.graph)

        # create some summaries
        with tf.name_scope('summaries'):

            # create all summaries
            tf.summary.scalar("mean_squared_error", self.error)
            tf.summary.scalar("absolute_error", self.a_error)
            tf.summary.scalar("learning_rate", self.learning_rate)

        # merge all summaries
        self.summaries = tf.summary.merge_all()

        # init if not restored
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def __init_all_cells(self):

        # create the common
        cell_config = {}
        cell_config['num_layers'] = self.config['rec_depth']
        cell_config['coupled_gates'] = self.config['rec_coupled_gates']
        cell_config['learn_hidden'] = self.config['rec_learnable_hidden_states']
        cell_config['h_activation'] = self.config['rec_h_node_activation']
        cell_config['num_input'] = self.config['num_input']
        cell_config['num_hidden'] = self.config['rec_num_hidden']
        cell_config['seed'] = self.config['seed']
        cell_config['training_time'] = self.training_time
        cell_config['zone_out_probability'] = self.config['zone_out_probability']
        cell_config['dropout_prob'] = self.config['dropout_prob']

        # init combined cells
        cells = list()
        for k in range(self.config['rec_num_stacks']):

            cell_config['cell_name'] = str(k)
            cell_config['head_of_stack'] = k == 0
            cells.append(RecurrentHighWayNetworkCell(cell_config))

        return cells

    def num_params(self):

        num = 0
        for cell in self.cells:
            num += cell.num_params()

        num += self.pre_highway_network.num_params()
        num += self.post_highway_network.num_params()

        return num

    def get_activation(self, name):
        if name == 'tanh':
            return tf.nn.tanh
        elif name == 'sigmoid':
            return tf.nn.sigmoid
        elif name == 'identity':
            return tf.identity
        elif name == 'lrelu':
            return lambda x: tf.maximum(x, 0.01 * x)

    def get_hidden_to_hidden_network(self, config, processed_unstacked_x, h_prev):

        # use for dynamic
        hidden_states = list()

        # unfold the cell
        for x_in in processed_unstacked_x:

            # create new h
            h = config['rec_num_stacks'] * [None]

            # create a cell
            h[0] = self.cells[0].create_cell(x_in, tf.nn.dropout(h_prev[0], config['dropout_prob']), [None])

            # stack them up
            for k in range(1, config['rec_num_stacks']):
                h[k] = self.cells[0].create_cell(x_in, h_prev[k], tf.nn.dropout(h[k - 1], config['dropout_prob']))

            # append to list
            hidden_states.append(h)
            h_prev = h

        return hidden_states

    def get_input_to_hidden_network(self, unstacked_x):
        """This method should take a list of input variables and
        process them through the same highway network.

        Args:
            unstacked_x: The list of input variables.

        Returns
            A list of processed variables.

        """
        return [self.pre_highway_network.get_graph(x_in) for x_in in unstacked_x]

    def get_hidden_to_output_network(self, lst_h, labels):
        """This method creates the hidden to output layer.

        Args:
            lst_h: A list of list, containing the hidden states for each layer.
            labels: List of Tensors containing the target labels

        Returns:
            The output layer itself.
        """

        # First of all guide the hidden states through the post highway network
        outputs = [self.post_highway_network.get_graph(tf.concat(h, axis=0)) for h in lst_h]

        # Check how much class outputs are there
        O = self.config['num_output']
        N = self.config['num_class_slots']
        label_loss = tf.constant(0.0, dtype=tf.float32)

        # When classification slots are activated
        pred_labels = list()
        classification_error = tf.constant(0.0, dtype=tf.float32)

        if N > 0:
            for k in range(len(outputs)):

                # create the cross entropy using the labels and the corresponding nodes
                squeezed_logits = tf.squeeze(outputs[k][O:, :])
                cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=squeezed_logits, labels=tf.cast(labels[k], tf.float32))

                # Add all single terms in cross entropy to label loss
                label_loss += tf.reduce_sum(cross_entropy)
                predicted_labels = tf.abs(tf.round(tf.sigmoid(outputs[k][O, :])) - tf.cast(labels[k], tf.float32))
                classification_error += tf.reduce_sum(predicted_labels)

                # Additionally transform
                outputs[k] = outputs[k][:O, :]
                pred_labels.append(predicted_labels)

        return outputs, pred_labels, classification_error, label_loss

    def get_preprocess_network(self):
        """This method delivers the pre processing network."""

        # create the configuration for the pre processing network
        highway_conf = {
            'name': 'preprocess_network',
            'num_input': self.config['num_input'],
            'num_hidden': self.config['pre_num_hidden'],
            'num_output': self.config['rec_num_hidden'],
            'num_layers': self.config['pre_num_layers'],
            'coupled_gates': self.config['pre_coupled_gates'],
            'in_activation': self.config['pre_in_activation'],
            'out_activation': self.config['pre_out_activation'],
            'h_node_activation': self.config['pre_h_node_activation'],
            'seed': self.config['seed'],
            'dropout_prob': self.config['dropout_prob'],
            'layer_normalization': self.config['pre_layer_normalization'],
            'num_class_slots': 0
        }

        return HighwayNetwork(highway_conf)

    def get_postprocess_network(self):
        """This method delivers the pre processing network."""

        # create the configuration for the pre processing network
        highway_conf = {
            'name': 'postprocess_network',
            'num_input': self.config['rec_num_stacks'] * self.config['rec_num_hidden'],
            'num_hidden': self.config['post_num_hidden'],
            'num_output': self.config['num_output'],
            'num_layers': self.config['post_num_layers'],
            'coupled_gates': self.config['post_coupled_gates'],
            'in_activation': self.config['post_in_activation'],
            'out_activation': self.config['post_out_activation'],
            'h_node_activation': self.config['post_h_node_activation'],
            'seed': self.config['seed'],
            'dropout_prob': self.config['dropout_prob'],
            'layer_normalization': self.config['post_layer_normalization'],
            'num_class_slots': self.config['num_class_slots']
        }

        return HighwayNetwork(highway_conf)

    def write_summaries(self, summary, test, episode):
        """Depending on the parameter either a test summary or train summary is written."""

        if test:
            self.test_writer.add_summary(summary, episode)

        else:
            self.train_writer.add_summary(summary, episode)

    def save(self, folder):
        """This method saves the model at the specified checkpoint."""
        all_vars = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        all_values = self.sess.run(all_vars)

        for k in range(len(all_values)):
            mod_name = str.replace(str.replace(all_vars[k].name, '/', '-'), '_', '-')
            np.save('{}{}/{}.npy'.format(self.config['log_dir'], folder, mod_name), all_values[k])

    def restore(self, folder):
        """This method restores the model from the specified folder."""
        all_vars = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        assign_list = list()
        for k in range(len(all_vars)):
            mod_name = str.replace(str.replace(all_vars[k].name, '/', '-'), '_', '-')
            tensor = np.load('{}{}/{}.npy'.format(self.config['log_dir'], folder, mod_name))
            assign_list.append(tf.assign(all_vars[k], tensor))

        self.sess.run(tf.group(*assign_list))

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
            trainer = minimizer.apply_gradients(zip(clipped_gradients, variables), global_step=global_step, name='minimizer')

        else:
            trainer = minimizer.apply_gradients(gradients, global_step=global_step, name='minimizer')

        return trainer

    def train(self, trajectories, target_trajectories, steps):
        """This method retrieves a list of trajectories. It can further
        process or transform these trajectories. But the desired overall
        behaviour is, that it should use the passed trajectories as training
        data and thus adjust the parameters of the model appropriately.

        Args:
            trajectories: This is a list of trajectories (A trajectory is a numpy vector)
            steps: The number of steps the model should execute.
        """

        fd = {self.x: trajectories, self.training_time: True,
              self.y: target_trajectories[:self.config['num_output'], :, :]}

        if self.config['num_class_slots'] > 0:
            fd[self.labels] = target_trajectories[self.config['num_output'], :, :]

        # we want to perform n steps
        for k in range(steps):
            _, summary = self.sess.run([self.minimizer, self.summaries], feed_dict=fd)

    def inc_episode(self):
        self.sess.run(self.inc_global_episode)

    def get_episode(self):
        return self.sess.run(self.global_episode)

    def validate(self, trajectories, target_trajectories, test=True, batch_size=131072):
        """This method basically validates the passed trajectories.
        It therefore splits them up so that the future frame get passed as the target.

        Args:
            trajectories: The trajectories to use for validation.
            target_trajectories: The target trajectories
            test: True, if validated on test error

        Returns:
            The error on the passed trajectories
        """

        num_trajectories = np.size(trajectories, 2)
        num_sets = int(np.floor((num_trajectories - 1) / batch_size))

        # create the errors
        overall_error = 0
        overall_single_error = 0
        overall_classification = 0
        s_div = 0
        a_div = 0
        c_div = 0

        # iterate over the number of sets
        for k in range(num_sets + 1):

            batch_trajectories = trajectories[:, :, k*batch_size:(k+1)*batch_size]
            batch_target_trajectories = target_trajectories[:self.config['num_output'], :, k*batch_size:(k+1)*batch_size]

            fd = {self.x: batch_trajectories, self.y: batch_target_trajectories, self.training_time: False}

            if self.config['num_class_slots'] > 0:
                fd[self.labels] = target_trajectories[self.config['num_output'], :, k*batch_size:(k+1)*batch_size]

            aerror, serror, class_error, summary, episode = self.sess.run([self.a_error, self.single_error, self.classification_rate, self.summaries, self.global_episode], feed_dict=fd)

            # add the errors
            s_mult = np.size(batch_trajectories, 2)
            a_mult = np.size(batch_trajectories, 1) * np.size(batch_trajectories, 2)

            overall_error += a_mult * aerror
            a_div += a_mult
            overall_single_error += s_mult * serror
            s_div += s_mult
            if self.config['num_class_slots'] > 0:
                c_mult = np.size(fd[self.labels])
                c_div += c_mult
                overall_classification += c_mult * class_error

            self.write_summaries(summary, test, episode)

        overall_error /= a_div
        overall_single_error /= s_div
        overall_classification /= c_div

        return overall_error, overall_single_error, overall_classification

    def predict(self, trajectories):
        return self.sess.run([self.target_y], feed_dict={self.x: trajectories, self.training_time: False})

    def init_params(self):
        """This initializes the parameters."""
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def reset(self):
        tf.reset_default_graph()
