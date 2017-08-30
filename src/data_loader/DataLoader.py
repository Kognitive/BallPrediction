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

from src.data_filter.DataFilter import DataFilter
from src.data_filter.concrete.IdentityDataFilter import IdentityDataFilter
from src.data_normalizer.DataNormalizer import DataNormalizer
from src.data_normalizer.concrete.IdentityNormalizer import IdentityNormalizer


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
        self.normalizer = IdentityNormalizer()
        self.loaded = False
        self.data = list()
        self.size = 0

    # This method has to be implemented to deliver some training data
    def load_data(self):
        raise NotImplementedError("You have to supply a training set.")


    # This method checks, if data was already loaded, if not it loads it.from
    # Finally it returns the completely loaded data.
    def load_complete_data(self):

        # if the data is not loaded already, reload it.
        if not self.loaded:

            print("Started to fetch data from HD")
            d = self.load_data()

            print("Data successfully loaded from HD")

            self.loaded = True
            self.data = [self.normalizer.normalize(x) for x in self.filter.filter(d)]
            #self.data = [np.random.randn(np.size(x, 0), np.size(x, 1)) * 0.01 + x for x in self.data]
            self.size = len(self.data)
            print("Normalized and filtered the data.")

        return self.data


    # This is a very inefficient implementation and should be overridden if possible
    def load_data(self, index):
        return self.load_complete_data()[index]


    # This method takes the passed data filter from the argument and
    # sets it inside in the filter attribute, so it can be used when
    # the data is loaded.
    #
    # data_filter - Pass the data filter, to use for the loader
    #
    def set_data_filter(self, data_filter):
        assert isinstance(data_filter, DataFilter)

        # set the filter and reset the data, because the new filter
        # has to be applied.
        self.filter = data_filter
        self.__invalidate_loaded_data()

    # This method takes a data normalizer and saves it internally.
    #
    # - data_normalizer: The data normalizer to use
    #
    def set_data_normalizer(self, data_normalizer):
        assert isinstance(data_normalizer, DataNormalizer)

        # set the normalizer and reset state of this object
        self.normalizer = data_normalizer
        self.__invalidate_loaded_data()

    # This method resets the state internally. In detail it
    # deletes the data from the object, so it has to be
    # reacquired by the loader.
    def __invalidate_loaded_data(self):
        self.loaded = False
        self.data = list()
