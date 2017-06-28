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

# This class can be used to create a queue, where you can insert elements, but
# retrieve the complete queue as one set.


class SetQueue:

    # This constructs a set queue.
    #
    # element_size - The size of one element
    # queue_size The size of the queue
    #
    def __init__(self, element_size, queue_size):

        # create a new empty vector
        self.index = 0
        self.count = 0
        self.data = np.zeros((queue_size, element_size))

        # save element and queue size internally for reuse
        self.ES = element_size
        self.QS = queue_size

    # This method inserts an element into the set queue.
    #
    # element - The element to insert inside of the queue.
    #
    def insert(self, element):

        # increase count by one, when the queue is not full now.
        if self.count < self.QS:
            self.count = self.count + 1

        # fill in the element
        self.data[self.index, :] = element
        self.index = (self.index + 1) % self.QS

    # This method delivers the whole set, if enough steps were made.
    def get(self):

        # divide into first and second half
        f = self.data[self.index:, :]
        s = self.data[:self.index, :]

        # pass back the result in the correct ordering
        return np.vstack([f, s])

    # This method resets the queue.
    def reset(self):

        # reset variables
        self.index = 0
        self.count = 0
