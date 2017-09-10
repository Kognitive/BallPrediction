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
import pickle

from os import listdir
from os.path import isfile, join, exists

from src.data_manager.DataManager import DataManager
from src.data_transformer.HiddenStateDataTransformer import HiddenStateDataTransformer


class SimHiddenDataManager(DataManager):
    """This class can be used to load the trajectories from the HD. The
    output format itself is a trajectory of x-y-z coordinates. The whole
    loader is designed in a lazy style."""

    def __init__(self, config):

        # define a counter
        super().__init__()

        # set boolean flag to false
        self.config = config

    def get_transformed_data(self, split):

        # load trajectories, split them up and transform them
        trajectories = self.load_complete_data()

        print("Splitting Data")
        path_count = len(trajectories)
        num_trajectories = path_count
        permutation = np.random.permutation(path_count)[:num_trajectories]
        num = int(np.ceil(num_trajectories / split))
        slices_va = permutation[:num]
        slices_tr = permutation[num:]

        # extract parameters
        I = self.config['I']
        K = self.config['K']

        print("Transforming Data")
        validation_set_in, validation_set_out = HiddenStateDataTransformer.transform([trajectories[i] for i in slices_va], I, K)
        training_set_in, training_set_out = HiddenStateDataTransformer.transform([trajectories[i] for i in slices_tr], I, K)

        return [validation_set_in, validation_set_out], [training_set_in, training_set_out]

    def load_data(self):

        # if we already have a pickle of the data just load it
        if exists(join(self.config['data_dir'], "data.pickle")):
            data = pickle.load(open(join(self.config['data_dir'], "data.pickle"), 'rb'))
            return data

        # get list of all subdirs
        subfiles = self.get_immediate_subfiles(self.config['data_dir'])

        # create empty positions array
        data = list()

        # iterate over the subdirs
        for file in subfiles:

            # access the loaded trajectory
            loaded_traj = np.loadtxt(file)[:, :]
            loaded_traj[:, 8] = (loaded_traj[:, 8] / 2) + 0.5

            # load the positions as well as the timestamp
            data.append(loaded_traj[:, :])

        pickle.dump(data, open(self.config['data_dir'] + "/data.pickle", 'wb'))
        return data

    # this method delivers all immediate subdirectories
    @staticmethod
    def get_immediate_subfiles(d):
        return [join(d, f) for f in listdir(d) if isfile(join(d, f))]
