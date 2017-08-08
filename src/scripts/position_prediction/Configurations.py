from src.models.concrete.LSTM import LSTM
from src.models.concrete.GatedRecurrentUnit import GatedRecurrentUnit
from src.models.concrete.RecurrentHighWayNetwork import RecurrentHighWayNetwork
from src.models.concrete.ClockworkRNN import ClockworkRNN

import tensorflow as tf

def lrelu(x): return tf.maximum(x, 0.01 * x)

class Configurations:
    """This class can be used to obtain the configuration along with the model."""


    @staticmethod
    def get_configuration_with_model(model_name):
        """This method takes a model_name as an input and returns
        the associated configuration as well as the model.

        Args:
            model_name: The name of the model

        Returns:
            Tuple (a,b) where a is the model and b the configuration
        """

        config = {}
        config['episodes'] = 100
        config['steps_per_episode'] = 10
        config['steps_per_batch'] = 1
        config['batch_size'] = 256
        config['num_input'] = 3
        config['num_output'] = 3

        if model_name == 'lstm':

            # create model and configuration
            model = LSTM
            config['unique_name'] = "StackLSTM"
            config['seed'] = 3

            config['num_hidden'] = 128
            config['num_layers'] = 24
            config['num_stacks'] = 5
            config['input_node_activation'] = tf.nn.tanh
            config['output_node_activation'] = lrelu
            config['peephole'] = False

            config['minimizer'] = 'adam'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.02
            config['lr_decay_steps'] = 100
            config['lr_decay_rate'] = 0.95

            config['layer_normalization'] = False
            config['clip_norm'] = 0

            config['preprocess_h_node_activation'] = tf.nn.tanh
            config['preprocess_activation'] = lrelu
            config['num_intermediate'] = 64
            config['num_preprocess_layers'] = 3
            config['preprocess_coupled_gates'] = True

        elif model_name == 'rhn':

            # create model and configuration
            model = RecurrentHighWayNetwork

            config['unique_name'] = "StackRecurrentHighwayNetwork"
            config['seed'] = 3

            config['num_hidden'] = 64
            config['num_layers'] = 24
            config['num_stacks'] = 5
            config['recurrence_depth'] = 10
            config['h_node_activation'] = tf.nn.tanh

            config['minimizer'] = 'adam'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.05
            config['lr_decay_steps'] = 100
            config['lr_decay_rate'] = 0.9

            config['coupled_gates'] = True
            config['layer_normalization'] = False
            config['clip_norm'] = 0

            config['preprocess_h_node_activation'] = tf.nn.tanh
            config['preprocess_activation'] = lrelu
            config['num_intermediate'] = 64
            config['num_preprocess_layers'] = 5
            config['preprocess_coupled_gates'] = True

        else:
            exit(1)

        return config, model
