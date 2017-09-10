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

from src.data_manager.concrete.SimDataLoader import SimDataLoader
from src.data_transformer.FeedForwardDataTransformer import FeedForwardDataTransformer
# we want to evaluate the LSTM
from src.models.concrete.LSTM import LSTM
from src.utils.Progressbar import Progressbar

data_dir = 'sim_training_data/data_v1'

# These are the fixed parameters
I = 3
episodes = 1000
trials = 250
line_length = 80
batch_size = 100

loader = SimDataLoader(data_dir)
trajectories = loader.load_complete_data()
path_count = len(trajectories)
permutation = np.random.permutation(path_count)
num = int(np.ceil(path_count / 2))
slices_va = permutation[:num]
slices_tr = permutation[num:]

# use the same validation and training set
V = [trajectories[i] for i in slices_va]
T = [trajectories[i] for i in slices_tr]

best_hyperparameters = 0
best_error = 10000000

# we want to write the results to a text file
f = open('run/hyper_parameter_search_lstm_rms_prop.txt', 'w')

# perform that much trials
for trial in range(trials):

    # These parameters we want to find trough a hyper parameter search
    hidden_size = int(np.floor(np.exp(np.random.uniform(np.log(5), np.log(30)))))
    cell_size = int(np.floor(np.exp(np.random.uniform(np.log(5), np.log(20)))))
    learning_rate = np.exp(np.random.uniform(np.log(0.00001), np.log(0.01)))
    momentum = 1 - np.exp(np.random.uniform(np.log(0.01), np.log(1.0)))
    unrolls = int(np.floor(np.exp(np.random.uniform(np.log(20), np.log(30)))))
    num_steps = np.random.randint(1, 5)

    hypers = (hidden_size, cell_size, learning_rate, momentum, unrolls, num_steps)
    print(line_length * "-")
    print("Trial " + str(trial))
    print(line_length * "-")
    print(str(hypers))

    # create the necessary objects
    transformer = FeedForwardDataTransformer(unrolls)
    model = LSTM(I, I, hidden_size, cell_size, unrolls, batch_size, learning_rate)
    model.init_params()

    # transform the data
    transformed_v = transformer.transform(V)
    transformed_tr = transformer.transform(T)


    pbar = Progressbar(episodes, line_length)

    # evaluate the task
    for episode in range(episodes):
        pbar.progress()
        model.train(transformed_tr, num_steps)

    # for each trial
    val_error = model.validate(transformed_v)
    print()
    print("Validation Error is: " + str(val_error))
    f.write(str(hypers) + " -> " + str(val_error) + "\n")
    f.flush()

    if val_error < best_error:
        best_hyperparameters = hypers
        best_error = val_error

    model.reset()

f.write(str(best_hyperparameters) + " -> " + str(best_error))
f.close()
