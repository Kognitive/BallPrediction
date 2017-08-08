from src.models.concrete.LSTM import LSTM
from src.models.concrete.GatedRecurrentUnit import GatedRecurrentUnit
from src.models.concrete.RecurrentHighWayNetwork import RecurrentHighWayNetwork
from src.models.concrete.ClockworkRNN import ClockworkRNN


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

        if model_name == 'lstm':

            # create model and configuration
            model = LSTM
            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 64
            config['num_layers'] = 24
            config['num_stacks'] = 5
            config['batch_size'] = 256
            config['seed'] = 3
            config['minimizer'] = 'rmsprop'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.001
            config['lr_decay_steps'] = 100
            config['lr_decay_rate'] = 0.90
            config['clip_norm'] = 0
            config['pre_process_structure'] = [64]
            config['peephole'] = True

        elif model_name == 'gru':

            # create model and configuration
            model = GatedRecurrentUnit

            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 128
            config['num_cells'] = 3
            config['num_layers'] = 30
            config['batch_size'] = 100
            config['seed'] = 3
            config['minimizer'] = 'momentum'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.0005
            config['lr_decay_steps'] = 1000
            config['lr_decay_rate'] = 0.9

        elif model_name == 'rhn':

            # create model and configuration
            model = RecurrentHighWayNetwork

            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 16
            config['num_cells'] = 3
            config['num_layers'] = 24
            config['batch_size'] = 256
            config['seed'] = 3
            config['minimizer'] = 'momentum'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.0005
            config['lr_decay_steps'] = 100000
            config['lr_decay_rate'] = 0.9
            config['recurrence_depth'] = 16

        elif model_name == 'cwrnn':

            # create model and configuration
            model = ClockworkRNN

            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 64
            config['num_cells'] = 3
            config['num_layers'] = 30
            config['batch_size'] = 128
            config['seed'] = 3
            config['minimizer'] = 'momentum'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.1
            config['lr_decay_steps'] = 1000
            config['lr_decay_rate'] = 0.9

            config['clip_norm'] = 10
            config['num_modules'] = 16
            config['module_size'] = 8

        else:
            exit(1)

        return config, model
