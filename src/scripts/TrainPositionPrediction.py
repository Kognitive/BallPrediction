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

from src.models.concrete.RecurrentHighWayNetwork import RecurrentHighWayNetwork
from src.scripts.ScriptRunner import ScriptRunner

# =============================================================================
# =============================================================================

run_config = {

    # ------------ General ----------------------------------------------------

    'root_dir': 'run/position_prediction',
    'reload_dir': None,
    'reload_checkpoint': 'general',
    'use_wizard': False,

    # ------------ Sets -------------------------------------------------------

    'ratios': [0.8, 0.2],
    'train_set_index': 0,
    'validation_set_index': 1,
    'set_labels': ['training', 'validation'],

    # ------------ Data -------------------------------------------------------

    'data_dir': 'training_data/sim/data_v4',
    'data_grouping': [['pos_error', 3]],
    'col_in': [0, 2, 4],
    'col_out': [0, 2, 4],

    # ------------ Plots ------------------------------------------------------

    'num_visualized_results': 3
}

# =============================================================================
# =============================================================================

model = RecurrentHighWayNetwork
model_config = {

    # ------------ Training ---------------------------------------------------

    'episodes': 10000,
    'steps_per_episode': 100,
    'steps_per_batch': 1,
    'batch_size': 4096,

    # ------------ Properties -------------------------------------------------

    'unique_name': "RecurrentHighWayNetwork",
    'seed': 28,

    # ------------ Data -------------------------------------------------------

    'num_input': len(run_config['col_in']),
    'num_output': len(run_config['col_out']),
    'distance_model': False,
    'add_variance': False,

    # ------------ Minimization -----------------------------------------------

    'lr_rate': 0.01,
    'lr_decay_steps': 100,
    'lr_decay_rate': 0.85,
    'clip_norm': 10,
    'momentum': 0.95,
    'minimizer': 'adam',

    # ------------ Regularization ---------------------------------------------

    'dropout_prob': 1.0,
    'zone_out_probability': 0.0,

    # ------------ Recurrent --------------------------------------------------

    'rec_num_hidden': 32,
    'rec_num_layers': 33,
    'rec_num_layers_student_forcing': 100,
    'rec_num_layers_teacher_forcing': 0,
    'rec_num_stacks': 4,
    'rec_depth': 6,
    'rec_h_node_activation': 'tanh',
    'rec_learnable_hidden_states': True,
    'rec_coupled_gates': True,
    'rec_layer_normalization': True,

    # ------------ Preprocess -------------------------------------------------

    'pre_num_hidden': 16,
    'pre_num_layers': 4,
    'pre_in_activation': 'lrelu',
    'pre_out_activation': 'lrelu',
    'pre_h_node_activation': 'tanh',
    'pre_coupled_gates': True,
    'pre_layer_normalization': True,

    # ------------ Postprocess ------------------------------------------------

    'post_num_hidden': 32,
    'post_num_layers': 2,
    'post_in_activation': 'lrelu',
    'post_out_activation': 'identity',
    'post_h_node_activation': 'tanh',
    'post_coupled_gates': True,
    'post_layer_normalization': True
}

# retrieve the model and the standard_configuration
ScriptRunner(run_config, [model_config, RecurrentHighWayNetwork]).run()
