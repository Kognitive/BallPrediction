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

from src.data_loader.concrete.SimHiddenDataLoader import SimHiddenDataLoader
from src.data_transformer.HiddenStateDataTransformer import HiddenStateDataTransformer
from src.scripts.hidden_state_prediction.Configurations import Configurations
from src.utils.Progressbar import Progressbar
from src.models.RecurrentNeuralNetwork import RecurrentNeuralNetwork

# Data Settings
data_dir = 'sim_training_data/data_v4'
log_dir = 'run/hidden_state_prediction'

loader = SimHiddenDataLoader(data_dir)
show_train_error = True

# Format settings
line_length = 80

# the reload and the last timestamp
reload = False
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
    conf['num_output'] = (1 if calc_hidden else 0) + (3 if calc_velocity else 0)
    conf['calc_hidden'] = calc_hidden
    conf['calc_velocity'] = calc_velocity

else:

    print("Restoring Configuration")

    # first of all restore the configuration
    config.read(output_dir + 'configuration.ini')
    conf = dict(config.items("Model"))

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

# define the overall error
episodes = conf['episodes']
validation_error = np.zeros((conf['num_output'], episodes))
train_error = np.zeros((conf['num_output'], episodes))

# some debug printing

# define the model using the parameters from top
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
fig_pred = plt.figure(1)

# create the array of subplots
num_traj = 4
pred_ax_arr = [None] * (num_traj * 2)

# create all subplots
for k in range(num_traj * 2):
    num = 2 * 100 + num_traj * 10 + (k + 1)
    pred_ax_arr[k] = fig_pred.add_subplot(num, projection='3d')

# sample some trajectories to display
val_display_slices = np.random.permutation(conf['va_size'])[:num_traj]
tr_display_slices = np.random.permutation(conf['tr_size'])[:num_traj]
plt.ion()

if calc_velocity and calc_hidden:

    ax_arr = 2 * [None]

    # noinspection PyUnboundLocalVariable
    ax_arr[0] = fig_error.add_subplot(121)
    ax_arr[1] = fig_error.add_subplot(122)

if reload: model.restore('general')
s_episode = model.get_episode()

# check if validation error gets better
best_val_episode = 0
best_val_error = 1000000000

best_tr_episode = 0
best_tr_error = 1000000000

if reload:
    validation_error = np.load(conf['log_dir'] + 'general/va_error.npy')
    train_error = np.load(conf['log_dir'] + 'general/tr_error.npy')

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
    _, validation_error[:, episode] = model.validate(validation_set_in, validation_set_out, True)
    _, train_error[:, episode] = model.validate(training_set_in, training_set_out, False)

    # save the model including the validation error and training error
    model.save('general')
    np.save(conf['log_dir'] + 'general/va_error.npy', validation_error)
    np.save(conf['log_dir'] + 'general/tr_error.npy', train_error)

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

    # draw the training error and validation error
    plt.figure(0)
    fig_error.suptitle("Error is: " + str(combined_val_error))

    calc_both = calc_hidden and calc_velocity
    if not calc_both:
        plt.clf()

    hid_array = ax_arr[0] if calc_both else plt
    if calc_hidden:

        # get index of hidden state
        ind = 3 if calc_both else 0

        # plot the failure
        hid_array.cla()
        hid_array.plot(np.linspace(0, episode, episode + 1), validation_error[ind, 0:episode + 1], color='r', label='Validation Error')
        hid_array.plot(np.linspace(0, episode, episode + 1), train_error[ind, 0:episode + 1], color='b', label='Training Error')
        hid_array.legend()

    vel_array = ax_arr[1] if calc_both else plt
    if calc_velocity:
        vel_array.cla()
        vel_array.plot(np.linspace(0, episode, episode + 1), np.sqrt(np.sum(np.power(validation_error[0:3, 0:episode + 1], 2), axis=0)), color='r', label='Validation Error')
        vel_array.plot(np.linspace(0, episode, episode + 1), np.sqrt(np.sum(np.power(train_error[0:3, 0:episode + 1], 2), axis=0)), color='b', label='Training Error')
        vel_array.legend()

    # display 4 predictions (2 x validation, 2 x training)
    trajectories_in = np.concatenate((validation_set_in[:, :, val_display_slices], training_set_in[:, :, tr_display_slices]), 2)
    result_out = np.concatenate((validation_set_out[:, :, val_display_slices], training_set_out[:, :, tr_display_slices]), 2)
    prediction_out = np.asarray(model.predict(trajectories_in))[0]

    # print the trajectories
    for i in range(num_traj * 2):
        pred_ax_arr[i].cla()

        # print the input trajectory
        x = trajectories_in[0, :conf['rec_num_layers'], i]
        y = trajectories_in[1, :conf['rec_num_layers'], i]
        z = trajectories_in[2, :conf['rec_num_layers'], i]
        pred_ax_arr[i].plot(x, y, z, label='Input', color='#1e9646')

        # print the predicted trajectory
        start_x = trajectories_in[0, -(conf['rec_num_layers_teacher_forcing'] + 1), i]
        start_y = trajectories_in[1, -(conf['rec_num_layers_teacher_forcing'] + 1), i]
        start_z = trajectories_in[2, -(conf['rec_num_layers_teacher_forcing'] + 1), i]
        dx = prediction_out[0, 0, i]
        dy = prediction_out[1, 0, i]
        dz = prediction_out[2, 0, i]
        comb_x = start_x + dx
        comb_y = start_y + dy
        comb_z = start_z + dz
        pred_ax_arr[i].plot([start_x, comb_x], [start_y, comb_y], [start_z, comb_z], label='Pred Velocity',
                            color='#1455bc')

        # Plot the real hidden state
        if calc_hidden:
            ind = 3 if calc_velocity else 0
            rz = result_out[ind, 0, i]
            pred_ax_arr[i].scatter(0, 0, rz, label='Real Hidden', color='#d15555a0')

            # print the real trajectory
            z = prediction_out[ind, 0, i]
            pred_ax_arr[i].scatter(0, 0, z, label='Pred Hidden', color='#870404')

        if calc_velocity:

            # print the predicted trajectory
            rdx = result_out[0, 0, i]
            rdy = result_out[1, 0, i]
            rdz = result_out[2, 0, i]
            rcomb_x = start_x + rdx
            rcomb_y = start_y + rdy
            rcomb_z = start_z + rdz
            pred_ax_arr[i].plot([start_x, rcomb_x], [start_y, rcomb_y], [start_z, rcomb_z], label='Real_Velocity', color='#81b0f9a0')



    plt.pause(0.01)

    # update progressbar
    p_bar.progress()

plt.ioff()
os.rename(output_dir, old_output_dir + str(validation_error[episodes - 1]))
