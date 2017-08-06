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
        if model_name == 'lstm':

            # create model and configuration
            model = LSTM

            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 20
            config['num_cells'] = 3
            config['num_layers'] = 20
            config['batch_size'] = 20
            config['seed'] = 3
            config['minimizer'] = 'momentum'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.01
            config['lr_decay_steps'] = 1000
            config['lr_decay_rate'] = 0.9
            config['peephole'] = True
            config['recurrence_depth'] = 6

            config['clip_norm'] = 15
            config['num_modules'] = 5
            config['module_size'] = 5

        elif model_name == 'gru':

            # create model and configuration
            model = GatedRecurrentUnit

            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 20
            config['num_cells'] = 3
            config['num_layers'] = 20
            config['batch_size'] = 20
            config['seed'] = 3
            config['minimizer'] = 'momentum'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.01
            config['lr_decay_steps'] = 1000
            config['lr_decay_rate'] = 0.9
            config['peephole'] = True
            config['recurrence_depth'] = 6

            config['clip_norm'] = 15
            config['num_modules'] = 5
            config['module_size'] = 5

        elif model_name == 'rhn':

            # create model and configuration
            model = RecurrentHighWayNetwork

            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 20
            config['num_cells'] = 3
            config['num_layers'] = 20
            config['batch_size'] = 20
            config['seed'] = 3
            config['minimizer'] = 'momentum'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.01
            config['lr_decay_steps'] = 1000
            config['lr_decay_rate'] = 0.9
            config['peephole'] = True
            config['recurrence_depth'] = 6

            config['clip_norm'] = 15
            config['num_modules'] = 5
            config['module_size'] = 5

        elif model_name == 'cwrnn':

            # create model and configuration
            model = ClockworkRNN

            config['unique_name'] = "1"
            config['num_input'] = 3
            config['num_output'] = 3
            config['num_hidden'] = 20
            config['num_cells'] = 3
            config['num_layers'] = 20
            config['batch_size'] = 20
            config['seed'] = 3
            config['minimizer'] = 'momentum'
            config['momentum'] = 0.95
            config['lr_rate'] = 0.01
            config['lr_decay_steps'] = 1000
            config['lr_decay_rate'] = 0.9
            config['peephole'] = True
            config['recurrence_depth'] = 6

            config['clip_norm'] = 15
            config['num_modules'] = 5
            config['module_size'] = 5