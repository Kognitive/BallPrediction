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

from src.data_adapter.DataAdapter import DataAdapter
from src.data_adapter.DataIterator import DataIterator

# This class represents an k offset adapter. It basically divides the
# trajectories into training examples for a pseudo recurrent neural
# network.
class KOffsetAdapter(DataAdapter, DataIterator):

    # this is the constructor for a data_adapter iterator. You have to supply it
    # via the file path to the root - containing all training samples.
    #
    #   I - Specify the size of the input, e.g. how many input frames
    #   O - Specify the size of the output, e.g. how many output frames
    #   K - Specify the temporal offset between input and output.
    #   root - The root folder for the data.
    #
    def __init__(self, I, O, K, root):

        # define a counter
        super().__init__()


        # save variables
        self.I = I
        self.O = O
        self.K = K

        # set boolean flag to false
        self.loaded = False
        self.root = root

        # generate the training data.
        self.gen_complete_training_data()

    # this method loads the data
    def load_data(self):

        # if already loaded return
        if self.loaded: return
        self.loaded = True

        # get list of all subdirs
        subdirs = self.get_immediate_subdirectories(self.root)

        # create empty positions array
        self.positions = list()

        # iterate over the subdirs
        for dir in subdirs:

            # load the positions as well as the timestamp
            self.positions.append(np.loadtxt(dir + "positions.txt"))

    # this method can be used to obtain the current data_adapter element.
    def obtain(self):

        # pass back a tuple containing the current training row
        return (self.positions[self.counter] + 5.0) / 10.0

    # this method should deliver the size of the data.
    def get_size(self):

        # simply deliver the size of the positions
        return len(self.positions)

    # this method delivers all immediate subdirectories
    def get_immediate_subdirectories(self, d):
        return glob(d + "/*/")

    # this method returns the input size
    def get_exact_input_size(self):
        return self.I * 3

    # this method returns the output size
    def get_exact_output_size(self):
        return self.O * 3

    # this method retrieves teh training data
    def get_complete_training_data(self):
        return self.cache

    # this method iterates over all training examples and returns all
    # combined in one numpy vector
    def gen_complete_training_data(self):

        # first of all reset, before iterating
        self.load_data()

        # get target and trainign pos
        training_data_pos = np.empty([self.I * 3, 0])
        target_data_pos = np.empty([self.O * 3, 0])
        count = 0
        queue_size = self.I + self.K

        # integrate
        tr_data_pos = list()
        ta_data_pos = list()

        # reset at start
        self.reset()

        # iterate over all positions
        while self.has_next():

            # count all samples
            trajectory = self.next()

            # continue path not good
            if (np.size(trajectory, 0) <= queue_size): continue

            # define the array, used as a queue
            array = np.empty([queue_size, 3])

            # fill the first I examples, with the next value
            for i in range(queue_size):
                array[i, :] = trajectory[i, :]

            # solange ein nächstes verfügbar ist
            num_points = np.size(trajectory, 0)

            # create tr and ta
            tr = np.empty([num_points - queue_size, self.I * 3])
            ta = np.empty([num_points - queue_size, self.O * 3])

            for el in range(queue_size, num_points):

                # get next sample
                n = trajectory[el, :]
                queue_index = el % queue_size

                # get the right and left indices
                right = (queue_index + self.I) % queue_size
                left = queue_index

                # when the left is bigger than zero, extract only one part
                # otherwise two
                if (left < right):
                    samples = np.reshape(array[left:right, :], self.I * 3)

                else:
                    stacked = np.vstack([array[left:, :], array[:right, :]])
                    samples = np.reshape(stacked, self.I * 3)

                # forward the array index
                array[queue_index, :] = n
                tr[el - queue_size, :] = samples
                ta[el - queue_size, :] = n

            # append to training data_adapter
            tr_data_pos.append(tr)
            ta_data_pos.append(ta)



        # pass pack all positions
        self.cache = [tr_data_pos, ta_data_pos]
