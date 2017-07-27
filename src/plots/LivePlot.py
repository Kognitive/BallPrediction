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
import matplotlib.pyplot as plt


class LivePlot:
    """This class represents a live plot, which is capable
    of updating data live on the screen."""

    def __init__(self):

        self.fig = plt.figure()

    def update_plot(self, episode, val_error, train_error=None):
        """This method updates the plot.

        Args:
            episode: Which episode are we in.
            val_error: The array of all errors.
            train_error: The array on all train errors
        """

        plt.clf()
        self.fig.suptitle("Error is: " + str(val_error[episode]))
        plt.plot(np.linspace(0, episode, episode + 1), val_error[0:episode+1], color='r', label='Validation Error')
        if train_error is not None:
            plt.plot(np.linspace(0, episode, episode + 1), train_error[0:episode+1], color='b', label='Training Error')

        plt.legend()
        plt.pause(0.01)

    def close(self):
        plt.ioff()
        plt.show()
