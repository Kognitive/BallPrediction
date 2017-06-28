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

from src.data_filter.DataFilter import DataFilter
from src.data_filter.concrete.IdentityDataFilter import IdentityDataFilter

# This class represents a DataLoader. It consists of some control variables and
# a cache. In order to use it inside of the software with a concrete
# implementation you have to give a load_data method, which basically returns
# a list of trajectories.


class DataLoader:

    # Basic initializer for the DataLoader
    def __init__(self):
        super().__init__()

        # init the vars
        self.filter = IdentityDataFilter()
        self.loaded = False
        self.data = list()

    # This method returns the size of the data set. So it basically tells
    # how many different trajectories are available
    def get_size(self):
        return len(self.data)

    # This method has to be implemented to deliver some training data
    def load_data(self):
        raise NotImplementedError("You have to supply a training set.")

    # This method checks, if data was already loaded, if not it loads it.from
    # Finally it returns the completely loaded data.
    def load_complete_data(self):

        # if the data is not loaded already, reload it.
        if not self.loaded:
            self.loaded = True
            self.data = list()

            # filter all trajectories
            for unfiltered in self.load_data():
                self.data.append(self.filter.filter(unfiltered))

        return self.data

    # This method takes the passed data filter from the argument and
    # sets it inside in the filter attribute, so it can be used when
    # the data is loaded.
    def set_data_filter(self, data_filter):
        assert isinstance(data_filter, DataFilter)

        # set the filter and reset the data, because the new filter
        # has to be applied.
        self.filter = data_filter
        self.loaded = False
        self.data = list()
