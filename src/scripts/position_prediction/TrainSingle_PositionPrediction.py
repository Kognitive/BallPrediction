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
from localconfig import config

# import live plot
from src.plots.LivePlot import LivePlot

from src.data_loader.concrete.SimDataLoader import SimDataLoader
from src.data_transformer.FeedForwardDataTransformer import FeedForwardDataTransformer
from src.scripts.position_prediction.Configurations import Configurations
from src.utils.Progressbar import Progressbar
from src.models.RecurrentNeuralNetwork import RecurrentNeuralNetwork

# Data Settings
data_dir = 'sim_training_data/data_v1'
log_dir = 'run/position_prediction'

loader = SimDataLoader(data_dir)
show_train_error = True

# Format settings
line_length = 80

# the reload and the last timestamp
reload = False
last_timestamp = "2017-08-29_00-01-15"

# ------------------------ SCRIPT -------------------------

# define the line length for printing
line = line_length * "-"
print(line)
print("Creating log directory")

# create the timestamp
timestamp = '{:%Y-%m-%d_%H-%M-%S}'.format(datetime.datetime.now()) if not reload else last_timestamp

old_output_dir = log_dir + "/"
output_dir = log_dir + "/" + timestamp + "_RUNNING/"
if not os.path.exists(output_dir):
    if reload: raise RuntimeError("This timestamp can't be reloaded")
    os.makedirs(output_dir)

if not os.path.exists(output_dir + "general"):
    if reload: raise RuntimeError("This timestamp can't be reloaded")
    os.makedirs(output_dir + "general")

# create the model folders
if not os.path.exists(output_dir + "mod_tr"):
    os.makedirs(output_dir + "mod_tr")

if not os.path.exists(output_dir + "mod_va"):
    os.makedirs(output_dir + "mod_va")

# retrieve the model
conf, chosen_model = Configurations.get_configuration_with_model('rhn')

# check if it should be reloaded or not
if not reload:

    conf['log_dir'] = output_dir
    conf['time_stamp'] = timestamp

else:

    print("Restoring Configuration")

    # first of all restore the configuration
    config.read(output_dir + 'configuration.ini')
    conf = dict(config.items("Model"))

# some debug printing
print("Creating Model")

# define the model using the parameters from top
model = chosen_model(conf)
model.init_params()

print(line)

# transformation parameters
I = conf['num_layers']
K = conf['num_layers_self']

# load trajectories, split them up and transform them
trajectories = loader.load_complete_data()
path_count = len(trajectories)
num_trajectories = path_count
permutation = np.random.permutation(path_count)[:num_trajectories]
num = int(np.ceil(num_trajectories / 5))
slices_va = permutation[:num]
slices_tr = permutation[num:]

# transform the data
validation_set_in, validation_set_out = FeedForwardDataTransformer.transform([trajectories[i] for i in slices_va], I, K)
training_set_in, training_set_out = FeedForwardDataTransformer.transform([trajectories[i] for i in slices_tr], I, K)

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
validation_error = np.zeros(episodes)
train_error = np.zeros(episodes)
overall_error = np.zeros([2, episodes])

# some debug printing
print("Model: " + model.name)
print(line)

# create progressbar
p_bar = Progressbar(episodes, line_length)

# check if validation error gets better
best_val_episode = 0
best_val_error = 1000000000

best_tr_episode = 0
best_tr_error = 1000000000

# create the figures appropriately
fig_error = plt.figure(0)
fig_pred = plt.figure(1)

# create the array of subplots
num_traj = 3
ax_arr = [None] * (num_traj * 2)

# create all subplots
for k in range(num_traj * 2):
    num = 2 * 100 + num_traj * 10 + (k + 1)
    ax_arr[k] = fig_pred.add_subplot(num, projection='3d')

# set interactive mode on
plt.ion()

# sample some trajectories to display
val_display_slices = np.random.permutation(conf['va_size'])[:num_traj]
tr_display_slices = np.random.permutation(conf['tr_size'])[:num_traj]

if reload: model.restore('general')
s_episode = model.get_episode()

if reload:
    validation_error[0:s_episode] = np.load(conf['log_dir'] + 'general/va_error.npy')
    train_error[0:s_episode] = np.load(conf['log_dir'] + 'general/tr_error.npy')

# set the range
for episode in range(s_episode, episodes):

    # execute as much episodes
    for step in range(conf['steps_per_episode']):

        # sample them randomly according to the batch size
        slices = np.random.randint(0, conf['tr_size'], conf['batch_size'])

        # train the model
        model.train(training_set_in[:, :, slices], training_set_out[:, :, slices], conf['steps_per_batch'])

    # increase the episode
    model.inc_episode()

    # calculate validation error
    validation_error[episode] = model.validate(validation_set_in, validation_set_out, True)
    train_error[episode] = model.validate(training_set_in, training_set_out, False)

    # save the model including the validation error and training error
    model.save('general')
    np.save(conf['log_dir'] + 'general/va_error.npy', validation_error[0:episode + 1])
    np.save(conf['log_dir'] + 'general/tr_error.npy', train_error[0:episode + 1])

    # check if we have to update our best error
    if best_val_error > validation_error[episode]:
        best_val_episode = episode
        best_val_error = validation_error[episode]
        model.save('mod_va')

        # check if we have to update our best error
    if best_tr_error > train_error[episode]:
        best_tr_episode = episode
        best_tr_error = train_error[episode]
        model.save('mod_tr')

    # draw the training error and validation error
    plt.figure(0)
    plt.clf()
    fig_error.suptitle("Error is: " + str(validation_error[episode]))
    plt.plot(np.linspace(0, episode, episode + 1), validation_error[0:episode + 1], color='r', label='Validation Error')
    plt.plot(np.linspace(0, episode, episode + 1), train_error[0:episode + 1], color='b', label='Training Error')
    plt.legend()

    # display 4 predictions (2 x validation, 2 x training)
    trajectories_in = np.concatenate((validation_set_in[:, :, val_display_slices], training_set_in[:, :, tr_display_slices]), 2)
    trajectories_out = np.concatenate((validation_set_out[:, :, val_display_slices], training_set_out[:, :, tr_display_slices]), 2)
    prediction_out = np.asarray(model.predict(trajectories_in))[0]

    # print the trajectories
    for i in range(num_traj * 2):

        ax_arr[i].cla()

        # print the real trajectory
        x = trajectories_out[0, :, i]
        y = trajectories_out[1, :, i]
        z = trajectories_out[2, :, i]
        ax_arr[i].plot(x, y, z, label='Real')

        # print the predicted trajectory
        x = prediction_out[0, :, i]
        y = prediction_out[1, :, i]
        z = prediction_out[2, :, i]
        ax_arr[i].plot(x, y, z, label='Prediction')

        # print the predicted trajectory
        x = trajectories_in[0, :, i]
        y = trajectories_in[1, :, i]
        z = trajectories_in[2, :, i]
        ax_arr[i].plot(x, y, z, label='Input')
        ax_arr[i].legend()

    plt.pause(0.01)

    # update progressbar
    p_bar.progress()

plt.ioff()
os.rename(output_dir, old_output_dir + str(validation_error[episodes - 1]))
