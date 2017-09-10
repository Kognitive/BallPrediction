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


class DataManager:
    """This class can be used to load matrix data stored in a column-wise manner.
    Additionally it will split it up and pass back the result."""

    def __init__(self, config):
        """Constructs a new DataManager. Can be used to load the data correctly.

        Args:
            config:
                col_in - A vector containing the input columns
                col_out - A vector containing the output columns
                offset: The offset between the first input of the input rows and the output rows
                ratios: Ratios on how the set should be divided
        """

        # define a counter
        super().__init__()

        # set boolean flag to false
        self.config = config
        self.loaded = False
        self.data = list()

    def load_divided_data(self):
        """This method is used to load and transform the input trajectories, so they
        can be processed further in the application.

        Args:
        """

        # check if it was loaded already
        if self.loaded:
            return self.data

        # if we already have a pickle of the data just load it
        if exists(join(self.config['data_dir'], "data.pickle")):
            self.data = pickle.load(open(join(self.config['data_dir'], "data.pickle"), 'rb'))
            return self.data

        # extract relevant parameters from the config
        col_in = self.config['col_in']
        col_out = self.config['col_out']
        ratios = self.config['ratios']
        in_length = self.config['in_length']
        out_length = self.config['out_length']

        # get list of all subdirs
        trajectory_files = self.get_immediate_subfiles(self.config['data_dir'], extension='.traj')
        trajectory_count = len(trajectory_files)
        assert trajectory_count > 0

        # get ratios for the trajectory num
        norm_ratios = ratios / np.sum(ratios)
        cum_ratios = np.cumsum(norm_ratios)
        absolute_indices = np.ndarray.astype(cum_ratios * trajectory_count, np.int32)
        absolute_indices[-1] = trajectory_count

        # create empty positions array
        fi = 0

        # iterate over the subdirs
        for si in absolute_indices:

            # create the data list
            in_data_list = list()
            out_data_list = list()

            # iterate over the indices selected
            for ti in range(fi, si):

                # obtain the file and load it
                file_name = trajectory_files[ti]
                trajectory = np.transpose(np.loadtxt(file_name))
                len_trajectory = np.size(trajectory, 1)

                # define the input and output trajectories
                for start in range(len_trajectory - (in_length + out_length + 1) + 1):

                    in_data_list.append(trajectory[col_in, start:start+in_length])
                    out_data_list.append(trajectory[col_out, start+in_length:start+in_length+out_length+1])

            # stack the data and collect in list
            in_data = np.stack(in_data_list, axis=2)
            out_data = np.stack(out_data_list, axis=2)
            data_tuple = (in_data, out_data)
            self.data.append(data_tuple)
            fi = si

        #pickle.dump(self.data, open(self.config['data_dir'] + "/data.pickle", 'wb'))
        return self.data

    @staticmethod
    def get_immediate_subfiles(d, extension):
        """Delivers the immediate subfiles inside of the directory
        using the specified extension.
        """

        return [join(d, f)
                for f in listdir(d)
                if isfile(join(d, f))
                and f.endswith(extension)]
