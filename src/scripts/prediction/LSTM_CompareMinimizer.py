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

import datetime
import os.path

# import the training controller
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.data_manager.concrete.SimDataLoader import SimDataLoader
from src.data_transformer.FeedForwardDataTransformer import FeedForwardDataTransformer
from src.models.concrete.LSTM import LSTM
from src.utils.Progressbar import Progressbar

# Data Settings
data_dir = 'sim_training_data/data_v1'
output_dir = 'run/lstm/minimizer_comparison'

# the loader
loader = SimDataLoader(data_dir)

# Format settings
line_length = 80

# Training parameters
learning_rate = 0.01
learning_threshold = 15
episodes = 1000
batch_size = 100
steps_per_episode = 100
steps_per_batch = 1
show_train_error = False
seed = 5

# Model settings
in_out_size = 3
unrolls = 20
hidden_cells = 10
memory_size = 10

# define the minimizers to use
minimizers = [["RMSProp", tf.train.RMSPropOptimizer, 'r'],
              ["Adam", tf.train.AdamOptimizer, 'b'],
              ["Adagrad", tf.train.AdagradOptimizer, 'g'],
              ["GradientDescent", tf.train.GradientDescentOptimizer, 'y']]

# ------------------------ SCRIPT -------------------------

# create the timestamp
timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())
output_dir = output_dir + "/" + timestamp + "/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# define the line length for printing
line = line_length * "-"
print(line)
print("Creating Model")

# extract number of models
num_models = len(minimizers)

# define the model using the parameters from top
models = [LSTM(x[0], in_out_size, in_out_size, memory_size, hidden_cells, unrolls, batch_size, x[1], seed)
          for x in minimizers]

for model in models: model.init_params()
print(line)

# create the transformer
transformer = FeedForwardDataTransformer(unrolls)

# load trajectories, split them up and transform them
trajectories = loader.load_complete_data()
path_count = len(trajectories)
permutation = np.random.permutation(path_count)
num = int(np.ceil(path_count / 3))
slices_va = permutation[:num]
slices_tr = permutation[num:]
validation_set = transformer.transform([trajectories[i] for i in slices_va])
training_set = transformer.transform([trajectories[i] for i in slices_tr])

# define the overall error
validation_error = np.zeros((num_models, episodes + 1))
train_error = np.zeros((num_models, episodes + 1))

model_string = "Model: LSTM(" + str(in_out_size) + ", " + str(in_out_size) + ", " + str(memory_size) + ", " \
              + str(hidden_cells) + ", " + str(unrolls) + ", " + str(batch_size) + ", " + str(seed) + ")"

# some debug printing
print(line)
print(model_string)
print(line)

# create progressbar
p_bar = Progressbar(episodes, line_length)

# check if validation error gets better
best_episode = np.ones(num_models)
best_val_error = np.ones(num_models)
learning_rates = learning_rate * np.ones(num_models)

# first validation error
for model in range(num_models):
    validation_error[model, 0] = models[model].validate(validation_set)

# create the figure
plt.figure()
plt.ion()

# set the range
for episode in range(1, episodes + 1):

    # iterate over all models
    for model in range(num_models):

        # execute as much episodes
        for step in range(steps_per_episode):

            # train the model
            models[model].train(training_set, steps_per_batch, learning_rates[model])

        # calculate validation error
        validation_error[model, episode] = models[model].validate(validation_set)
        if show_train_error: train_error[model, episode] = models[model].validate(training_set)

        # check if we have to update our best error
        if best_val_error[model] > validation_error[model, episode]:
            best_episode[model] = episode
            best_val_error[model] = validation_error[model, episode]

        # when the offset between the current is to high
        if episode - best_episode[model] > learning_threshold:
            learning_rates[model] /= 5

    # clear everything from the plot
        plt.clf()

    # plot every model
    for model in range(num_models):

        plt.title(model_string)
        plt.plot(np.linspace(0, episode, episode + 1), validation_error[model, 0:episode+1],
                 color=minimizers[model][2],
                 label=(minimizers[model][0] + " (" + str(round(validation_error[model, episode], 10)) + ")"))

        # plt.axhline(y=validation_error[model, episode], color=minimizers[model][2], linestyle=':')
        # plt.text(validation_error[model, episode] + 0.2, 0, str(validation_error[model, episode]))

    plt.legend()
    plt.pause(0.01)
    plt.savefig(output_dir + 'comparison.eps')

    # update progressbar
    p_bar.progress()

plt.show()
