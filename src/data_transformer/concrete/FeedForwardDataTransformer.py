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
    def __init__(self, I, O, K):
        self.I = I
        self.O = O
        self.K = K
        self.QS = self.I + self.O + self.K

    # This method has to be implemented in order to support the data transformation.
    #
    # trajectories - the list of trajectories to transform.
    #
    def transform(self, trajectories):

        # create the arrays
        training_data_pos = np.empty([self.I * 3, 0])
        target_data_pos = np.empty([self.O * 3, 0])

        # integrate
        tr_data_pos = list()
        ta_data_pos = list()

        # create a new state queue
        state_queue = SetQueue(3, self.QS)

        # iterate over all trajectories
        for trajectory in trajectories:

            # continue when size to small
            if np.size(trajectory, 0) <= self.QS:
                continue

            # reset the queue
            state_queue.reset()

            # fill queue
            for i in range(self.QS - 1):
                state_queue.insert(trajectory[i, :])

            num_points = np.size(trajectory, 0)

            # create tr and ta
            tr = np.empty([num_points - self.QS, self.I * 3])
            ta = np.empty([num_points - self.QS, self.O * 3])

            # iterate over the remaining points
            for i in range(self.QS - 1, num_points):

                # insert into queue
                state_queue.insert(trajectory[i, :])
                data = state_queue.get()

                # get input and output
                tr[i - self.QS, :] = np.reshape(data[:self.I, :], self.I * 3)
                ta[i - self.QS, :] = np.reshape(data[(self.I + self.K):, :], self.O * 3)

            # append to training data_loader
            tr_data_pos.append(tr)
            ta_data_pos.append(ta)

        return [np.transpose(np.vstack(tr_data_pos)), np.transpose(np.vstack(ta_data_pos))]