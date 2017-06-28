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

from scipy.signal import butter, lfilter
from src.data_filter.DataFilter import DataFilter

# This class represent a identity data filter. So the output is the same as the
# input.


class LowPassFilter(DataFilter):

    def __init(self, cutoff=1.2, fs=120.0, order=6):
        self.cutoff = cutoff
        self.fs = fs
        self.order = order

    # Return the trajectory.
    def filter(self, trajectories):

        ftrajectories = [[self.__applyFilter(trajectories[i][j]) for j in range(len(trajectories[i]))]
                 for i in range(len(trajectories))]
        ftrajectories = [list(filter(lambda x: x.shape[0] != 0, ftrajectories[i])) for i in range(len(ftrajectories))]

        return ftrajectories

    def __applyFilter(self, data):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = np.zeros(data.shape)
        for i in range(data.shape[1]):
            filtered_data[:,i] = lfilter(b, a, data[:,i])
        # Cut off parts of the begin and of the end because low pass filters tend to produce unusable output at the ends
        return filtered_data[int(self.fs):len(filtered_data) - int(self.fs / 2),:]