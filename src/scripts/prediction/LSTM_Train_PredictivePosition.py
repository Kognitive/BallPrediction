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

# import the training controller
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.controller.concrete.PathFoldController import PathFoldController
from src.data_loader.concrete.SimDataLoader import SimDataLoader
from src.models.concrete.LSTM import LSTM
from src.models.concrete.GatedRecurrentUnit import GatedRecurrentUnit
from src.models.concrete.RecurrentHighWayNetwork import RecurrentHighWayNetwork
from src.utils.Progressbar import Progressbar
from src.plots.LivePlot import LivePlot
from src.data_transformer.concrete.FeedForwardDataTransformer import FeedForwardDataTransformer

# Data Settings
data_dir = 'sim_training_data/data_v1'
loader = SimDataLoader(data_dir)

# Format settings
line_length = 80

# Training parameters
episodes = 1000
batch_size = 100
steps_per_episode = 100
steps_per_batch = 1
show_plots = True

# ------------------------ SCRIPT -------------------------

# define the line length for printing
line = line_length * "-"

# create the configuration
choosen_model = RecurrentHighWayNetwork
config = {}
config['unique_name'] = "1"
config['num_input'] = 3
config['num_output'] = 3
config['num_hidden'] = 20
config['num_cells'] = 3
config['num_layers'] = 20
config['batch_size'] = batch_size
config['seed'] = 3
config['minimizer'] = 'momentum'
config['momentum'] = 0.95
config['lr_rate'] = 0.01
config['lr_decay_steps'] = 1000
config['lr_decay_rate'] = 0.9
config['peephole'] = True
config['recurrence_depth'] = 6

# create the transformer
transformer = FeedForwardDataTransformer(config['num_layers'])

# load trajectories, split them up and transform them
trajectories = loader.load_complete_data()
path_count = len(trajectories)
permutation = np.random.permutation(path_count)
num = int(np.ceil(path_count / 3))
slices_va = permutation[:num]
slices_tr = permutation[num:]
validation_set = transformer.transform([trajectories[i] for i in slices_va])
training_set = transformer.transform([trajectories[i] for i in slices_tr])

# define the size of training and validation set
config['tr_size'] = np.size(training_set, 2)
config['va_size'] = np.size(validation_set, 2)

print(line)
print("Creating Model")
# define the model using the parameters from top
model = choosen_model(config)
model.init_params()

# upload the data
print(line)

# define the overall error
validation_error = np.zeros(episodes)
train_error = np.zeros(episodes)
overall_error = np.zeros([2, episodes])

# some debug printing
print("Model: " + model.name)
print(line)

# create progressbar
p_bar = Progressbar(episodes, line_length)
lv = LivePlot()

# check if validation error gets better
best_episode = 0
best_val_error = 1000000000

# set the range
for episode in range(episodes):

    # execute as much episodes
    for step in range(steps_per_episode):

        # train the model
        model.train(training_set, steps_per_batch)

    # calculate validation error
    validation_error[episode] = model.validate(validation_set)
    train_error[episode] = model.validate(training_set)

    # check if we have to update our best error
    if best_val_error > validation_error[episode]:
        best_episode = episode
        best_val_error = validation_error[episode]

    # update progressbar
    p_bar.progress()
    lv.update_plot(episode, validation_error, train_error)

lv.close()