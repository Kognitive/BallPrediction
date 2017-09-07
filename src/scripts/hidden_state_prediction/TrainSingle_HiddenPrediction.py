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
no_user_input = True

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

# upload the data
print(line)
print("Splitting Validation Data")

# transform the data
validation_set_in, validation_set_out = HiddenStateDataTransformer.transform([trajectories[i] for i in slices_va], I, K)
print("Splitted Validation Data")
print("Splitting Training Data")
training_set_in, training_set_out = HiddenStateDataTransformer.transform([trajectories[i] for i in slices_tr], I, K)
print("Splitted Training Data")

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

# define the overall error
episodes = conf['episodes']
validation_error = np.zeros((conf['num_output'], episodes))
validation_class_error = np.zeros(episodes)
train_error = np.zeros((conf['num_output'], episodes))
train_class_error = np.zeros(episodes)

# some debug printing

# define the model using the parameters from top

# some debug printing
print("Creating Model")

model = chosen_model(conf)
model.init_params()
conf['model_params'] = model.num_params()

# some debug printing
print(line)
print("Model {} ({})".format(model.name, conf['model_params']))
print(line)

# create progressbar
p_bar = Progressbar(episodes, line_length)

# create both figures
fig_error = plt.figure(0)
err_axes = [None] * 3
err_axes[0] = fig_error.add_subplot(131)
err_axes[1] = fig_error.add_subplot(132)
err_axes[2] = fig_error.add_subplot(133)
plt.ion()

if reload: model.restore('general')
s_episode = model.get_episode()

# check if validation error gets better
best_val_episode = 0
best_val_error = 1000000000

best_tr_episode = 0
best_tr_error = 1000000000
combined_val_error = 100
combined_tr_error = 100

if reload:
    train_error = np.load(conf['log_dir'] + 'general/tr_error.npy')
    train_class_error = np.load(conf['log_dir'] + 'general/tr_class_error.npy')
    validation_error = np.load(conf['log_dir'] + 'general/va_error.npy')
    validation_class_error = np.load(conf['log_dir'] + 'general/va_class_error.npy')

    # check if validation error gets better
    mean_val = np.mean(validation_error, axis=0)[0:s_episode]
    best_val_episode = np.argmin(mean_val)
    best_val_error = np.min(mean_val)

    # training error as well
    mean_tra = np.mean(train_error, axis=0)[0:s_episode]
    best_tr_episode = np.argmin(mean_tra)
    best_tr_error = np.min(mean_tra)

    if reload_val:
        s_episode = best_val_episode + 1

update = False
semaphore = threading.Semaphore()
saved_episode = 0

def train():
    global model, conf, s_episode, p_bar
    global validation_class_error, validation_error, train_class_error, train_error
    global best_val_episode, best_val_error, best_tr_episode, best_tr_error
    global training_set_in, training_set_out, validation_set_in, validation_set_out
    global combined_val_error, combined_tr_error
    global semaphore, update
    global episode, saved_episode

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
        _, validation_error[:, episode], validation_class_error[episode] = model.validate(validation_set_in, validation_set_out, True)
        _, train_error[:, episode], train_class_error[episode] = model.validate(training_set_in, training_set_out, False)

        # update plot data
        semaphore.acquire()

        update = True
        saved_episode = episode

        semaphore.release()

        # save the model including the validation error and training error
        model.save('general')
        np.save(conf['log_dir'] + 'general/va_error.npy', validation_error)
        np.save(conf['log_dir'] + 'general/va_class_error.npy', validation_class_error)
        np.save(conf['log_dir'] + 'general/tr_error.npy', train_error)
        np.save(conf['log_dir'] + 'general/tr_class_error.npy', train_class_error)

        # get the combined error
        combined_val_error = np.mean(validation_error[:, episode])
        combined_tr_error = np.mean(train_error[:, episode])

        # check if we have to update our best error
        if best_val_error > combined_val_error:
            best_val_episode = episode
            best_val_error = combined_val_error
            model.save('mod_va')

            # check if we have to update our best error
        if best_tr_error > combined_tr_error:
            best_tr_episode = episode
            best_tr_error = combined_tr_error
            model.save('mod_tr')

    # update progressbar
    p_bar.progress()


def plot():
    global model_statistics, trajectory_plot
    global semaphore, update
    global combined_val_error, combined_tr_error
    global model, conf, s_episode, p_bar
    global validation_class_error, validation_error, train_class_error, train_error
    global best_val_episode, best_val_error, best_tr_episode, best_tr_error
    global training_set_in, training_set_out, validation_set_in, validation_set_out
    global combined_val_error, combined_tr_error
    global semaphore, update
    global episode, saved_episode

    semaphore.acquire()
    if update:

        # draw the training error and validation error
        plt.figure(0)
        fig_error.suptitle("Error is: " + str(combined_val_error))

        # plot the failure
        err_axes[0].cla()
        err_axes[0].plot(np.linspace(0, saved_episode, saved_episode + 1), validation_error[3, 0:saved_episode + 1], color='r', label='Validation Error')
        err_axes[0].plot(np.linspace(0, saved_episode, saved_episode + 1), train_error[3, 0:saved_episode + 1], color='b', label='Training Error')
        err_axes[0].legend()

        err_axes[1].cla()
        err_axes[1].plot(np.linspace(0, saved_episode, saved_episode + 1),
                           np.sqrt(np.sum(np.power(validation_error[0:3, 0:saved_episode + 1], 2), axis=0)), color='r',
                           label='Validation Error')
        err_axes[1].plot(np.linspace(0, saved_episode, saved_episode + 1),
                           np.sqrt(np.sum(np.power(train_error[0:3, 0:saved_episode + 1], 2), axis=0)), color='b',
                           label='Training Error')
        err_axes[1].legend()

        va_error = validation_class_error[0:saved_episode + 1]
        tr_error = train_class_error[0:saved_episode + 1]
        lin_s = np.linspace(0, saved_episode, saved_episode + 1)

        err_axes[2].cla()
        err_axes[2].plot(lin_s, va_error, color='r',
                         label='Validation Error')
        err_axes[2].plot(lin_s, tr_error, color='b',
                         label='Training Error')
        err_axes[2].legend()
        plt.pause(0.01)
        plt.show()

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


# os.rename(output_dir, old_output_dir + str(validation_error))