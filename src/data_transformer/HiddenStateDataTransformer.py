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

from src.utils.SetQueue import SetQueue


class HiddenStateDataTransformer:
    """This class can be used to transform data."""

    @staticmethod
    def transform(trajectories, I, K):
        """This method is used to transform the input trajectories, so they
        can be processed further in the application.

        Args:
            trajectories: The trajectories
            I: The number of input frames.
            K: The offset between the last input and the first output
        """

        # integrate
        input_data = list()
        target_data = list()

        # get the queue size
        queue_size = I + K

        # create a new state queue
        state_queue = SetQueue(3, queue_size)

        # iterate over all trajectories
        for trajectory in trajectories:

            # continue when size to small
            if np.size(trajectory, 0) <= queue_size:
                continue

            # reset the queue
            state_queue.reset()

            # fill complete queue with entries
            for i in range(queue_size - 1):
                state_queue.insert(trajectory[i, 0:6:2])

            # gather the overall number of points
            num_points = np.size(trajectory, 0)

            # iterate over the remaining points
            for i in range(queue_size - 1, num_points):

                # insert into queue
                state_queue.insert(trajectory[i, [0, 2, 4]])
                data = state_queue.get()
                in_data = data
                out_data = trajectory[(i-K):(i+1), [1, 3, 5, 6, 8]]
                input_data.append(np.transpose(in_data))
                target_data.append(np.transpose(out_data))

        return np.stack(input_data, axis=2), np.stack(target_data, axis=2)
