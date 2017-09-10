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


class StatManager:

    def __init__(self, episodes: int, root: str, files: dict, reload: bool, labels: list):

        self.stats_dict = {}
        self.output_dir = root
        self.file_count = len(files)
        self.files = files
        self.labels = labels

        for [file, size] in files:

            val_list = list()
            for prefix in labels:
                jfile = "{}_{}.npy".format(prefix, file)
                complete_file = join(root, jfile)

                # Determine whether to reload the data or to create it
                if reload:
                    print("Restoring {}".format(jfile))
                    val_list.append(np.load(complete_file))

                else:
                    print("Creating {}".format(jfile))
                    val_list.append(np.zeros((size, episodes)))

            self.stats_dict[file] = val_list

        # save a list of the keys
        self.last_episode = -1

        # Create the axes for the plot
        plt_error = plt.figure('Error')
        self.plt_axes = [None] * self.file_count

        # create all axes
        for i in range(self.file_count):
            num = self.file_count * 100 + 10 + i + 1
            self.plt_axes[i] = plt_error.add_subplot(num)

    def store_statistics(self, current_episode, errors):

        for error_index in range(len(errors)):
            error = errors[error_index]

            fi = 0
            for [file, si] in self.files:
                self.stats_dict[file][error_index][:, current_episode] = error[fi:fi+si]
                fi += si

        self.last_episode = current_episode

    def save_statistics(self):

        for [file, _] in self.files:
            for li in range(len(self.labels)):
                label = self.labels[li]
                complete_file = join(self.output_dir, "{}_{}".format(label, file))
                np.save(complete_file, self.stats_dict[file][li])

    def plot(self):

        # iterate over the files
        for i in range(self.file_count):
            key = self.files[i][0]

            # clear axes
            ax = self.plt_axes[i]
            ax.cla()

            # plot for all sets
            for li in range(len(self.labels)):
                m = np.mean(self.stats_dict[key][li][:, :self.last_episode + 1], axis=0)
                ax.plot(m, label=self.labels[li])

            ax.title.set_text('{} is {}'.format(key, m[-1]))
            ax.legend()
