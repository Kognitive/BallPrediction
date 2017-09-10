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
import matplotlib.pyplot as plt
import threading

from src.manager.DataManager import DataManager
from src.manager.RunManager import RunManager
from src.utils.Progressbar import Progressbar
from src.utils.ThreadedPause import ThreadedPause


class ScriptRunner:

    def __init__(self, run_config, model_details):
        self.run_config = run_config
        self.model_details = model_details

    def run(self):

        run_config = self.run_config
        model_details = self.model_details

        # Format settings
        line_length = 80
        line = line_length * "-"
        print(line)

        # create the log directory helper
        self.run_manager = RunManager(run_config)
        print(line)

        # Initialize the model
        self.run_manager.init_model(model_details)
        self.stats = self.run_manager.create_stats()
        self.conf = self.run_manager.model_config
        print(line)

        # transformation parameters
        run_config['in_length'] = self.run_manager.model_config['rec_num_layers']
        run_config['out_length'] = self.run_manager.model_config['rec_num_layers_teacher_forcing'] \
                                   + self.run_manager.model_config['rec_num_layers_student_forcing']

        # create the data manager and get the transformed data
        loader = DataManager(run_config)
        self.data_sets = loader.load_divided_data()
        set_labels = run_config['set_labels']

        # define the size of training and validation set
        for set_index in range(len(self.data_sets)):
            data = self.data_sets[set_index]
            size = np.size(data[0], 2)
            label = set_labels[set_index]
            print("Size of Set {} is {}".format(size, label))

        print(line)

        # create model header
        print("Model {} ({})".format(self.run_manager.model.name, self.conf['model_params']))
        print(line)

        # create progressbar
        self.p_bar = Progressbar(self.conf['episodes'], line_length)

        # status variables
        self.update = False
        self.semaphore = threading.Semaphore()
        self.progress_count = self.run_manager.current_episode

        # set interactive mode on
        plt.ion()
        plt.show()

        thread = threading.Thread(None, self.train, "Training Thread")
        thread.start()
        while thread.is_alive():
            self.plot()
            ThreadedPause.pause(2)

    def train(self):

        validation_set_index = self.run_config['validation_set_index']
        train_set_index = self.run_config['train_set_index']
        train_data = self.data_sets[train_set_index]

        # set the range
        for episode in range(self.run_manager.current_episode, self.conf['episodes']):

            # execute as much episodes
            for step in range(self.conf['steps_per_episode']):
                # sample them randomly according to the batch size and train the model
                slices = np.random.randint(0, np.size(train_data[0], 2), self.conf['batch_size'])
                self.run_manager.model.train(train_data[0][:, :, slices], train_data[1][:, :, slices],
                                             self.conf['steps_per_batch'])

            # calculate validation error
            all_errors = [self.run_manager.model.validate(data_set[0], data_set[1]) for data_set in self.data_sets]

            # update plot data
            self.semaphore.acquire()

            # update everything
            self.progress_count += 1
            self.update = True
            self.stats.store_statistics(episode, all_errors)

            self.semaphore.release()

            # get the combined error
            combined_val_error = np.mean(all_errors[validation_set_index])
            self.run_manager.inc_episode()

            # check if we have to update our best error
            if self.run_manager.best_value > combined_val_error or self.run_manager.best_episode == -1:
                self.run_manager.best_episode = episode
                self.run_manager.best_value = combined_val_error
                self.run_manager.save('best')

            # increase the episode
            self.run_manager.save('general')
            self.stats.save_statistics()

    def plot(self):

        self.semaphore.acquire()
        if self.update:
            for k in range(self.progress_count):
                self.p_bar.progress()

            self.progress_count = 0
            self.stats.plot()
            self.update = False

        self.semaphore.release()