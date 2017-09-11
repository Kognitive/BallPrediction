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
from mpl_toolkits.mplot3d import Axes3D

class PlotManager:

    def __init__(self, num_traj, in_grouping, out_grouping, input_data, real_output, pred_output):

        self.num_trajectories = num_traj
        self.in_grouping = in_grouping
        self.out_grouping = out_grouping

        self.fig = plt.figure(1)
        self.ax_arr = [None] * (num_traj * 2)

        # create all subplots
        for k in range(num_traj * 2):
            num = 2 * 100 + num_traj * 10 + (k + 1)
            self.ax_arr[k] = self.fig.add_subplot(num, projection='3d')

        self.fig.show()
        self.input = input_data
        self.real_output = real_output
        self.pred_output = pred_output

    def update_data(self, pred_output):

        self.pred_output = pred_output

    def plot(self):

        # print as much trajectories as wanted
        for pi in range(2 * self.num_trajectories):
            self.ax_arr[pi].cla()

            fi = 0
            last_input = 3 * [0]
            for [label, n, typ] in self.in_grouping:

                # find the second index
                si = fi + n

                # if it is of type trajectory
                if typ == 'traj':
                    assert n == 3
                    xyz = self.input[fi:si, :, pi]
                    self.ax_arr[pi].scatter(xyz[0], xyz[1], xyz[2], label=label)
                    self.ax_arr[pi].plot(xyz[0], xyz[1], xyz[2])
                    last_input = np.asarray([xyz[i, -1] for i in range(0, n)])

                fi = si

            fi = 0
            # go for the output as well
            for [label, n, typ] in self.out_grouping:

                # find the second index
                si = fi + n

                for [output, name] in [[self.real_output, 'real'], [self.pred_output, 'pred']]:

                    # if it is of type trajectory
                    if typ == 'traj':
                        assert n == 3
                        xyz = output[fi:si, :, pi]
                        self.ax_arr[pi].scatter(xyz[0], xyz[1], xyz[2], label="{}_{}".format(name, label))
                        self.ax_arr[pi].plot(xyz[0], xyz[1], xyz[2])

                    # if it is of type trajectory
                    elif typ == 'vector':
                        assert n == 3
                        xyz = np.stack([last_input, output[fi, 0, pi] + last_input], axis=1)
                        self.ax_arr[pi].plot(xyz[0], xyz[1], xyz[2], label="{}_{}".format(name, label))

                    elif typ == 'zpoint':
                        assert n == 1
                        x = 0
                        y = 0
                        z = output[fi, 0, pi]
                        self.ax_arr[pi].scatter(x, y, z, label="{}_{}".format(name, label))

                fi = si
            self.ax_arr[pi].legend()