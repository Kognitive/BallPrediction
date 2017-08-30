from src.models.concrete.LSTM import LSTM
from src.models.concrete.GatedRecurrentUnit import GatedRecurrentUnit
from src.models.concrete.RecurrentHighWayNetwork import RecurrentHighWayNetwork
from src.models.concrete.ClockworkRNN import ClockworkRNN

import tensorflow as tf

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

        # training details
        config['episodes'] = 40000
        config['steps_per_episode'] = 100
        config['steps_per_batch'] = 1
        config['batch_size'] = 4096

        # model
        model = RecurrentHighWayNetwork

        # basic properties of a model
        config['unique_name'] = "RecurrentHighWayNetwork"
        config['seed'] = 3
        config['distance_model'] = False

        # data details
        config['num_input'] = 3
        config['num_output'] = 3

        # minimizer settings (Adam doesn't use the parameters)
        config['minimizer'] = 'adam'
        config['momentum'] = 0.95
        config['lr_rate'] = 0.01
        config['lr_decay_steps'] = 100
        config['lr_decay_rate'] = 0.85
        config['clip_norm'] = 10

        # regularization parameters
        config['dropout_prob'] = 1.0
        config['zone_out_probability'] = 0.0

        # the settings for the recursive part
        config['rec_num_hidden'] = 32
        config['rec_num_layers'] = 10
        config['rec_num_layers_student_forcing'] = 25
        config['rec_num_layers_teacher_forcing'] = 0
        config['rec_num_stacks'] = 3
        config['rec_depth'] = 6
        config['rec_h_node_activation'] = 'tanh'
        config['rec_learnable_hidden_states'] = True
        config['rec_coupled_gates'] = True
        config['rec_layer_normalization'] = True

        # the settings for the preprocess network
        config['pre_num_hidden'] = 9
        config['pre_num_layers'] = 3
        config['pre_in_activation'] = 'lrelu'
        config['pre_out_activation'] = 'lrelu'
        config['pre_h_node_activation'] = 'tanh'
        config['pre_coupled_gates'] = True
        config['pre_layer_normalization'] = True

        # the settings for the postprocess network
        config['post_num_hidden'] = 12
        config['post_num_layers'] = 2
        config['post_in_activation'] = 'lrelu'
        config['post_out_activation'] = 'identity'
        config['post_h_node_activation'] = 'tanh'
        config['post_coupled_gates'] = True
        config['post_layer_normalization'] = True

        return config, model
