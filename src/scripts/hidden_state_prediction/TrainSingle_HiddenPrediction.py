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
import datetime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import configparser
import threading

from localconfig import config

# import live plot
from src.plots.LivePlot import LivePlot

from src.data_loader.concrete.SimHiddenDataLoader import SimHiddenDataLoader
from src.data_transformer.HiddenStateDataTransformer import HiddenStateDataTransformer
from src.scripts.hidden_state_prediction.Configurations import Configurations
from src.utils.Progressbar import Progressbar
from src.models.RecurrentNeuralNetwork import RecurrentNeuralNetwork
from src.utils.ModelStatistics import ModelStatistics
from src.utils.LogDirectoryHelper import LogDirectoryHelper
from src.utils.ThreadedPause import ThreadedPause
from src.utils.TrajectoryPlot import TrajectoryPlot

# Data Settings
data_dir = 'sim_training_data/data_v4'
log_dir = 'run/hidden_state_prediction'

loader = SimHiddenDataLoader(data_dir)

# Set to true if you want to run this script without user interaction
# This will automatically create the log directory as a new directory with the current timestamp as name
no_user_input = False

# Format settings
line_length = 80

# the reload and the last timestamp
reload_val = False
last_timestamp = "2017-09-02_03-02-02"
calc_velocity = True
calc_hidden = True

# ------------------------ SCRIPT -------------------------

# define the line length for printing
line = line_length * "-"
print(line)
print("Creating log directory")

# create the timestamp
timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now())

old_output_dir = log_dir + "/"

output_dir, reload = LogDirectoryHelper.create(log_dir, timestamp, no_user_input)

# retrieve the model
conf, chosen_model = Configurations.get_configuration_with_model('rhn')

# check if it should be reloaded or not
if not reload:
    conf['log_dir'] = output_dir
    conf['time_stamp'] = timestamp
    conf['num_output'] = (1 if calc_hidden else 0) + (3 if calc_velocity else 0)
    conf['calc_hidden'] = calc_hidden
    conf['calc_velocity'] = calc_velocity

else:
    print("Restoring Configuration")

    # first of all restore the configuration
    config.read(output_dir + 'configuration.ini')
    conf = dict(config.items("Model"))

# some debug printing
print("Creating Model")

print(line)

# transformation parameters
I = conf['rec_num_layers']
K = conf['rec_num_layers_teacher_forcing']

# load trajectories, split them up and transform them
trajectories = loader.load_complete_data()
path_count = len(trajectories)
num_trajectories = path_count
permutation = np.random.permutation(path_count)[:num_trajectories]
num = int(np.ceil(num_trajectories / 5))
slices_va = permutation[:num]
slices_tr = permutation[num:]

print("Splitting Validation Data")

# transform the data
validation_set_in, validation_set_out = HiddenStateDataTransformer.transform([trajectories[i] for i in slices_va], I, K)
print("Splitted Validation Data")
print("Splitting Training Data")
training_set_in, training_set_out = HiddenStateDataTransformer.transform([trajectories[i] for i in slices_tr], I, K)
print("Splitted Training Data")

assert calc_velocity or calc_hidden

if calc_velocity and not calc_hidden:
    validation_set_out = validation_set_out[0:3, :, :]
    training_set_out = training_set_out[0:3, :, :]

if calc_hidden and not calc_velocity:
    validation_set_out = validation_set_out[3:4, :, :]
    training_set_out = training_set_out[3:4, :, :]

# define the size of training and validation set
conf['tr_size'] = np.size(training_set_in, 2)
conf['va_size'] = np.size(validation_set_in, 2)

# write the config
parser = configparser.ConfigParser()
parser.add_section("Model")

# add all configuration parameters
for key in conf.keys():
    parser.set("Model", key, str(conf[key]))

# and write to disk
with open(output_dir + "configuration.ini", 'w') as cfg_file:
    parser.write(cfg_file)

print("Model configuration saved on disk.")

# upload the data
print(line)

# some debug printing

# define the model using the parameters from top
model = chosen_model(conf)
model.init_params()
conf['model_params'] = model.num_params()
model_statistics = ModelStatistics()

# Create the trajectory plot
num_visualized_trajectories = 3
trajectory_plot = TrajectoryPlot(num_visualized_trajectories)

# some debug printing
print(line)
print("Model {} ({})".format(model.name, conf['model_params']))
print(line)

# create progressbar
p_bar = Progressbar(conf['episodes'], line_length)

# sample some trajectories to display
val_display_slices = np.random.permutation(conf['va_size'])[:num_visualized_trajectories]
tr_display_slices = np.random.permutation(conf['tr_size'])[:num_visualized_trajectories]

if reload: model.restore('general')
s_episode = model.get_episode()

if reload:
    train_error_single = np.load(conf['log_dir'] + 'general/tr_error.npy')
    validation_error_single = np.load(conf['log_dir'] + 'general/va_error.npy')
    model_statistics.init(train_error_single, validation_error_single)

semaphore = threading.Semaphore()
update = False


def train():
    global model, conf, s_episode, model_statistics, trajectory_plot, p_bar
    global training_set_in, training_set_out, validation_set_in, validation_set_out
    global val_display_slices, tr_display_slices
    global semaphore, update
    # set the range
    for episode in range(s_episode, conf['episodes']):
        # execute as much episodes
        for step in range(conf['steps_per_episode']):
            # sample them randomly according to the batch size
            slices = np.random.randint(0, conf['tr_size'], conf['batch_size'])
            # train the model
            model.train(training_set_in[:, :, slices], training_set_out[:, :, slices], conf['steps_per_batch'])

        # increase the episode
        model.inc_episode()

        # calculate validation error
        validation_error, validation_error_single = model.validate(validation_set_in, validation_set_out, True)
        train_error, train_error_single = model.validate(training_set_in, training_set_out, False)

        # calculate 6 predictions for visualization (3 x validation, 3 x training)
        trajectories_in = np.concatenate((validation_set_in[:, :, val_display_slices],
                                          training_set_in[:, :, tr_display_slices]), 2)
        trajectories_out = np.concatenate((validation_set_out[:, :, val_display_slices],
                                           training_set_out[:, :, tr_display_slices]), 2)
        prediction_out = np.asarray(model.predict(trajectories_in))[0]

        # update plot data
        semaphore.acquire()

        model_statistics.update(train_error, train_error_single, validation_error, validation_error_single)
        trajectory_plot.update_trajectories(trajectories_in, trajectories_out, prediction_out)

        update = True

        semaphore.release()

        # save the model including the validation error and training error
        model.save('general')
        np.save(conf['log_dir'] + 'general/va_error.npy', validation_error)
        np.save(conf['log_dir'] + 'general/tr_error.npy', train_error)

        best_validation_episode, best_training_episode = model_statistics.get_best_episode()

        # Save model if it has the best validation or training error
        if best_validation_episode == episode:
            model.save('mod_va')
        if best_training_episode == episode:
            model.save('mod_tr')

        # update progressbar
        p_bar.progress()

    os.rename(output_dir, old_output_dir + str(model.validation_error[-1]))


def plot():
    global model_statistics, trajectory_plot
    global semaphore, update
    semaphore.acquire()
    if update:
        # plot the training error, validation error and the trajectories
        model_statistics.plot()
        trajectory_plot.plot()

        update = False
    semaphore.release()

if __name__ == '__main__':
    # set interactive mode on
    plt.ion()

    thread = threading.Thread(None, train, "Training Thread")
    thread.start()
    while thread.is_alive():
        plot()
        ThreadedPause.pause(2)
>>>>>>> 77bf723585f46a10ad2629713b15f122dd4aa10e
