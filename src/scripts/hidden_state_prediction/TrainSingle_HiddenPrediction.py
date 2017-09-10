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

import threading

import matplotlib.pyplot as plt
# import the training controller
import numpy as np
import os
import sys

from src.data_manager.concrete.SimHiddenDataManager import SimHiddenDataManager
from src.run_manager.RunManager import RunManager
from src.utils.Progressbar import Progressbar
from src.utils.ThreadedPause import ThreadedPause
from src.run_manager.HiddenTrajectoryPlot import HiddenTrajectoryPlot

# ------------------------ Configuration ------------------------

# Define the configuration for the run
run_config = {
    'root_dir': 'run/hidden_state_prediction',
    'reload_dir': None,
    'reload_checkpoint': 'general',
    'use_wizard': False,
    'model': 'rhn',
    'files': {'vel_error': 3, 'hidden_error': 1}
}

data_config = {
    'data_dir': 'sim_training_data/data_v5',
}

# Format settings
line_length = 80
line = line_length * "-"
print(line)

# ------------------------ Script -------------------------

# create the log directory helper
run_manager = RunManager(run_config)
print(line)

# transformation parameters
data_config['I'] = run_manager.model_config['rec_num_layers']
data_config['K'] = run_manager.model_config['rec_num_layers_teacher_forcing']

# create the data manager and get the transformed data
loader = SimHiddenDataManager(data_config)
val_data, tr_data = loader.get_transformed_data(5)

# define the size of training and validation set
print("Size of Training Set is: {}".format(np.size(tr_data[0], 2)))
print("Size of Validation Set is: {}".format(np.size(val_data[0], 2)))
print(line)

# Initialize the model
run_manager.init_model()
stats = run_manager.create_stats()
conf = run_manager.model_config

print(line)

# Create Model header
print("Model {} ({})".format(run_manager.model.name, conf['model_params']))
print(line)

# create progressbar
p_bar = Progressbar(conf['episodes'], line_length)

update = False
semaphore = threading.Semaphore()
progress_count = 0

def train():
    global run_manager, stats, conf
    global val_data, tr_data
    global semaphore, update
    global progress_count

    # set the range
    for episode in range(run_manager.current_episode, conf['episodes']):

        # execute as much episodes
        for step in range(conf['steps_per_episode']):

            # sample them randomly according to the batch size and train the model
            slices = np.random.randint(0, np.size(tr_data[0], 2), conf['batch_size'])
            run_manager.model.train(tr_data[0][:, :, slices], tr_data[1][:, :, slices], conf['steps_per_batch'])

        # calculate validation error
        _, validation_error = run_manager.model.validate(val_data[0], val_data[1], True)
        _, train_error = run_manager.model.validate(tr_data[0], tr_data[1], False)

        # update plot data
        semaphore.acquire()
        progress_count += 1
        update = True

        # store the statistics
        stats.store_statistics(episode, 'vel_error', train_error[0:3], validation_error[0:3])
        stats.store_statistics(episode, 'hidden_error', train_error[3], validation_error[3])
        # stats.store_statistics(episode, 'class_error', train_class_error, validation_class_error)

        semaphore.release()

        # get the combined error
        combined_val_error = np.mean(validation_error)
        run_manager.inc_episode()

        # check if we have to update our best error
        if run_manager.best_value > combined_val_error or run_manager.best_episode == -1:
            run_manager.best_episode = episode
            run_manager.best_value = combined_val_error
            run_manager.save('best')

        # increase the episode
        run_manager.save('general')
        stats.save_statistics()


def plot():
    global semaphore, update
    global stats
    global progress_count
    global p_bar

    semaphore.acquire()
    if update:
        for k in range(progress_count):
            p_bar.progress()
        progress_count = 0
        stats.plot()
        update = False

    semaphore.release()

if __name__ == '__main__':

    # set interactive mode on
    plt.ion()
    plt.show()

    thread = threading.Thread(None, train, "Training Thread")
    thread.start()
    while thread.is_alive():
        plot()
        ThreadedPause.pause(2)