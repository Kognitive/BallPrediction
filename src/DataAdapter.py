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
import scipy.ndimage as sp

# this class can be used to create a data iterator. This so called data
# adapter can be used to specify a location for training data. The interface
# of the iterator can then be used to iterate over the files.
class DataAdapter:

    # this is the constructor for a data iterator. You have to supply it
    # via the file path to the root - containing all training samples.
    def __init__(self, root):

        # define a counter
        self.data_counter = 0

        # load the positions as well as the timestamp
        self.time_stamps = np.loadtxt(root + "/timestamps.txt")
        self.positions = np.loadtxt(root + "/positions.txt")

    # this method can be used to obtain the current data element.
    def obtain(self):

        # basically pass back the image
        img = sp.imread(str(self.data_counter + 1) + ".png")

        # pass back a tuple containing the current training row
        return [self.time_stamps[self.data_counter], self.positions[self.data_counter], img]

    # this method defines the next. It actually increases the
    # data counter by one and then calls the internally obtain mehtod
    def next(self):

        self.data_counter = self.data_counter + 1
        return self.obtain()

    # this method returns the input size
    def get_input_size(self):
        return 10

    # this method returns the output size
    def get_output_size(self):
        return 10


    # this method iterates over all training examples and returns all
    # combined in one numpy vector
    def get_all(self):
        return NotImplementedError("Implemtn please")
