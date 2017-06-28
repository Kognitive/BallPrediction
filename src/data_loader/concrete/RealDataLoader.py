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

from glob import glob
from src.data_loader.DataLoader import DataLoader

# This class can be used to load the trajectories from the HD. The
# output format itself is a trajectory of x-y-z coordinates. The whole
# loader is designed in a lazy style.


class RealDataLoader(DataLoader):

    # this is the constructor for a simulation training data adapter
    #
    #   root - The root folder for the data.
    #
    def __init__(self, root):

        # define a counter
        super().__init__()

        # set boolean flag to false
        self.root = root

    # this method loads the data
    def load_data(self):

        # get list of all subdirs
        subdirs = self.get_immediate_subdirectories(self.root)

        # create empty positions array
        data = list()

        # iterate over the subdirs
        for dir in subdirs:

            # load the positions as well as the timestamp
            data.append(np.loadtxt(dir + "positions.txt"))

        return data

    # this method delivers all immediate subdirectories
    @staticmethod
    def get_immediate_subdirectories(d):
        return glob(d + "/*/")
