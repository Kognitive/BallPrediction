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

from src.data_adapter.DataIterator import DataIterator
# this class represents a data adapter, you have to extend this
# class in order to use it with a training controller.
class DataAdapter(DataIterator):

    def __init__(self):
        super().__init__()
        self.index = -1
        self.data_loaded = False

    # this method has to be implemented and deliver some training examples
    # from the storage
    def get_complete_training_data(self):
        raise NotImplementedError("You have to supply a training set.")

    def __initialiseIterator(self):
        if self.data_loaded:
            return
        self.data = self.get_complete_training_data()
        self.data_loaded = True

    def obtain(self):
        self.__initialiseIterator()
        if self.index >= 0 and self.index < len(self.data):
            return self.data[self.index]
        raise IndexError("No item to obtain at current position")

    def get_size(self):
        self.__initialiseIterator()
        return len(self.data)

