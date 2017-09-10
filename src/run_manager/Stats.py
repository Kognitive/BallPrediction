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

from os.path import join

class Stats:

    def __init__(self, episodes: int, root: str, files: dict, reload: bool):

        self.stats_dict = {}
        self.output_dir = root
        self.file_count = len(files)

        for file in files:

            val_list = list()
            for prefix in ['tr', 'va']:
                jfile = "{}_{}.npy".format(prefix, file)
                complete_file = join(root, jfile)

                # Determine whether to reload the data or to create it
                if reload:
                    print("Restoring {}.npy".format(jfile))
                    val_list.append(np.load(complete_file))

                else:
                    print("Creating {}.npy".format(jfile))
                    val_list.append(np.zeros((files[file], episodes)))

            self.stats_dict[file] = val_list

        # save a list of the keys
        self.keys = list(files.keys())
        self.last_episode = -1

        # Create the axes for the plot
        plt_error = plt.figure('Error')
        self.plt_axes = [None] * self.file_count

        # create all axes
        for i in range(self.file_count):
            num = self.file_count * 100 + 10 + i + 1
            self.plt_axes[i] = plt_error.add_subplot(num)

    def store_statistics(self, current_episode, file, tr_value, va_value):
        self.stats_dict[file][0][:, current_episode] = tr_value
        self.stats_dict[file][1][:, current_episode] = va_value
        self.last_episode = current_episode

    def save_statistics(self):

        prefix_list = ['tr', 'va']
        for file in self.stats_dict:
            for i in range(2):
                complete_file = join(self.output_dir, "{}_{}".format(prefix_list[i], file))
                np.save(complete_file, self.stats_dict[file][i])

    def plot(self):

        # iterate over the files
        for i in range(self.file_count):
            key = self.keys[i]

            # clear axes
            ax = self.plt_axes[i]
            ax.cla()

            # get va and tr
            tr = np.mean(self.stats_dict[key][0][:, :self.last_episode + 1], axis=0)
            va = np.mean(self.stats_dict[key][1][:, :self.last_episode + 1], axis=0)

            ax.title.set_text('{} is {}'.format(key, va[-1]))
            ax.plot(va, color='r', label="Validation")
            ax.plot(tr, color='b', label="Training")
            ax.legend()