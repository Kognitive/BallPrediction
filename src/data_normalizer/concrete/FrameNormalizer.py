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

from src.data_normalizer.DataNormalizer import DataNormalizer


# Represents a frame normalizer
class FrameNormalizer(DataNormalizer):

    # Constructs a new FrameNormalizer. It therefore
    # receives an input and and output dimensions.
    #
    # - In: A 2-element list stating the input interval
    # - Out: A 2-element list stating the output interval
    #
    def __init__(self, inp, outp):

        # Get a1 and a2 respectively
        self.a1 = inp[0]
        self.a2 = outp[0]
        self.M = (outp[1] - outp[0]) / (inp[1] - inp[0])

    # Perform bijective mapping from input interval to output
    # interval.
    #
    # - data: The data to use.
    #
    def normalize(self, data):

        # normalize
        return self.M * (data - self.a1) + self.a2

    # The normalize mapping is bijective and thus the inverse
    # exists and we can un_normalize the data with this method.
    #
    # - data: The data to un normalize.
    #
    def un_normalize(self, data):

        # un normalize
        return (data - self.a2) / self.M + self.a1
