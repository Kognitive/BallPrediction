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

    # constructs a low pass filter.
    #
    # - W: Threshold, where to cut off the frequencies
    # - F: frames per second
    # - N: order of the filter
    #
    def __init__(self, W=1.2, F=120.0, N=6):
        self.W = W
        self.F = F
        self.N = N

    def apply_filter(self, data):

        # Normalize by Nyquist frequency
        nyq = 0.5 * self.F
        normal_cutoff = self.W / nyq

        # apply the Butterworth filter
        b, a = butter(self.N, normal_cutoff, btype='low', analog=False)
        filtered_data = np.empty(data.shape)

        # Iterate over the vertical dimension
        for i in range(data.shape[1]):
            filtered_data[:, i] = lfilter(b, a, data[:, i])

        # Cut off parts of the begin and of the end because low pass filters tend to produce unusable output at the ends
        return filtered_data[int(self.F):len(filtered_data) - int(self.F / 2), :]
