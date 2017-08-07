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
import numpy as np

from src.data_loader.concrete.SimDataLoader import SimDataLoader
from src.data_transformer.FeedForwardDataTransformer import FeedForwardDataTransformer
from src.plots.LivePlot import LivePlot
from src.scripts.position_prediction.Configurations import Configurations
from src.utils.Progressbar import Progressbar

# Data Settings
data_dir = 'sim_training_data/data_v1'
output_dir = 'run/lstm/minimizer_comparison'
loader = SimDataLoader(data_dir)
show_train_error = True

# Format settings
line_length = 80

# ------------------------ SCRIPT -------------------------

# define the line length for printing
line = line_length * "-"
print(line)

# retrieve the model
config, chosen_model = Configurations.get_configuration_with_model('rhn')

# load trajectories, split them up and transform them
trajectories = loader.load_complete_data()
path_count = len(trajectories)
num_trajectories = path_count
permutation = np.random.permutation(path_count)[:num_trajectories]
num = int(np.ceil(num_trajectories / 3))
slices_va = permutation[:num]
slices_tr = permutation[num:]

# transformation parameters
I = config['num_layers']
K = 20

# transform the data
validation_set_in, validation_set_out = FeedForwardDataTransformer.transform([trajectories[i] for i in slices_va], I, K)
training_set_in, training_set_out = FeedForwardDataTransformer.transform([trajectories[i] for i in slices_tr], I, K)

# define the size of training and validation set
config['tr_size'] = np.size(training_set_in, 2)
config['va_size'] = np.size(validation_set_in, 2)

# some debug printing
print(line)
print("Creating Model")

# define the model using the parameters from top
model = chosen_model(config)
model.init_params()

# upload the data
print(line)

# define the overall error
episodes = config['episodes']
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
    for step in range(config['steps_per_episode']):

        # sample them randomly according to the batch size
        slices = np.random.randint(0, config['tr_size'], config['batch_size'])

        # train the model
        model.train(training_set_in[:, :, slices], training_set_out[:, slices], config['steps_per_batch'])

    # calculate validation error
    validation_error[episode] = model.validate(validation_set_in, validation_set_out)
    if show_train_error: train_error[episode] = model.validate(training_set_in, training_set_out)

    # check if we have to update our best error
    if best_val_error > validation_error[episode]:
        best_episode = episode
        best_val_error = validation_error[episode]

    # update progressbar
    p_bar.progress()
    lv.update_plot(episode, validation_error, train_error if show_train_error else None)

lv.save('run/rhn.eps')
lv.close()