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

# this class represents a data iterator. It can be used to iterate over data.
# which is stored on the harddrive. This comes in handy, when you want to
# implement batch like training for neural networks
class DataIterator:

    # simply set the counter to -1, so you can iterate over
    def __init__(self):
        self.reset()

    # this method can be used to obtain the current data_adapter element.
    def obtain(self):
        raise NotImplementedError("You have to make data samples obtainable.")

    # this method should deliver the size of the data.
    def get_size(self):
        raise NotImplementedError("You have to supply a size.")

    # check if it has an return appropriately next
    def next(self):
        if not self.has_next():
            raise IndexError("Please check in advance if there is a next element.")

        self.counter = self.counter + 1
        return self.obtain()

    # checks if there is a next one
    def has_next(self):
        return self.counter < self.get_size() - 1

    # this method resets the counter
    def reset(self):
        self.counter = -1