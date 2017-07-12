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

from src.data_transformer.DataTransformer import DataTransformer
from src.utils.SetQueue import SetQueue

# This class is a basic FeedForwardDataTransformer. It basically transforms trajectories
# into supervised training data.


class FeedForwardDataTransformer(DataTransformer):

    # This constructs the FeedForwardDataTransformer
    #
    # I - the number of position inputs
    # O - the number of position outputs
    # K - the offset between input and output
    #
    def __init__(self, I):

        # save the hyper parameters
        self.I = I

        # this is the maximum size
        self.QS = self.I + 1

    # This method has to be implemented in order to support the data transformation.
    #
    # trajectories - the list of trajectories to transform.
    #
    def transform(self, trajectories):

        # integrate
        input_data = list()

        # create a new state queue
        state_queue = SetQueue(3, self.QS)

        # iterate over all trajectories
        for trajectory in trajectories:

            # continue when size to small
            if np.size(trajectory, 0) <= self.QS:
                continue

            # reset the queue
            state_queue.reset()

            # fill complete queue with entries
            for i in range(self.QS - 1):
                state_queue.insert(trajectory[i, :])

            # gather the overall number of points
            num_points = np.size(trajectory, 0)

            # iterate over the remaining points
            for i in range(self.QS - 1, num_points):

                # insert into queue
                state_queue.insert(trajectory[i, :])
                data = state_queue.get()
                input_data.append(np.transpose(data))

        return np.stack(input_data, axis=2)
