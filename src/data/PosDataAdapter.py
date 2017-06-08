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
from queue import Queue

import numpy as np
import scipy.ndimage as sp

# this class can be used to create a data iterator. This so called data
# adapter can be used to specify a location for training data. The interface
# of the iterator can then be used to iterate over the files.
class PosDataAdapter:

    # this is the constructor for a data iterator. You have to supply it
    # via the file path to the root - containing all training samples.
    def __init__(self, root):

        # define a counter
        self.data_counter = -1

        # load the positions as well as the timestamp
        self.positions = np.loadtxt(root + "/positions.txt")

    # this method can be used to obtain the current data element.
    def obtain(self):

        # pass back a tuple containing the current training row
        return (self.positions[self.data_counter] + 5.0) / 10.0

    # checks if there is a next one
    def has_next(self):
        return self.data_counter < np.size(self.positions, axis = 0) - 1

    # this method resets the counter
    def reset(self):
        self.data_counter = 0

    # this method defines the next. It actually increases the
    # data counter by one and then calls the internally obtain mehtod
    def next(self):

        self.data_counter = self.data_counter + 1
        return self.obtain()

    # this method returns the input size
    def get_input_size(self):
        return 90

    # this method returns the output size
    def get_output_size(self):
        return 3

    # this method iterates over all training examples and returns all
    # combined in one numpy vector
    def get_all(self):

        self.reset()
        offset = 100
        cOffset = offset
        maxs = 30

        array = np.empty([3, maxs + offset])
        array_ind = 0
        for i in range(10):
            array[:, i] = self.next()

        # integrate
        array_ind = maxs

        # get target and trainign pos
        training_data_pos = np.empty([maxs * 3, 0])
        target_data_pos = np.empty([3, 0])

        # solange ein nächstes verfügbar ist
        while self.has_next():
            n = self.next()

            # check whether the offset is zero, in order to
            # append a training example to the internal structure.
            if (cOffset == 0):

                # get the right and left indices
                right = (array_ind - offset + 1 + maxs + offset) % (maxs + offset)
                left = right - maxs
                if (left >= 0):
                    samples = np.reshape(np.transpose(array[:, left:right]), (right - left) * 3)
                else:

                    stacked = np.hstack([array[:, left:], array[:, 0:right]])
                    samples = np.reshape(np.transpose(stacked), (right - left) * 3)

                # append to training data
                input_ex = np.expand_dims(samples, 1)
                training_data_pos = np.hstack([training_data_pos, input_ex])
                target_data_pos = np.hstack([target_data_pos, np.expand_dims(n[:], 1)])

            else:
                cOffset = cOffset - 1

            # forward the array index
            array[:, array_ind] = n
            array_ind = (array_ind + 1) % (maxs + offset)

        # pass pack all positions
        return [training_data_pos, target_data_pos]
