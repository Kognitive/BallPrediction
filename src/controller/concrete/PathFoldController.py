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

# import the necessary packages
import numpy as np

# own packages
from src.controller.TrainingController import TrainingController
from src.models.PredictionModel import PredictionModel
from src.data_adapter.DataAdapter import DataAdapter

# this class is a basic controller
from src.utils.Progressbar import Progressbar


class ProgressBar(object):
    pass


class PathFoldController(TrainingController):

    # this constructor creates a new data_adapter iterator
    # and saves the passed prediction model
    #
    #   adapter - A data adpater, which is capable of supplying the data.
    #   model - choose a model wisely
    #   F - which fold should be selected
    #   N - how many folds are there overall
    #
    def __init__(self, adapter, model, F, N):

        # check whether the prediction model is an instance of the
        # correct interface
        assert isinstance(model, PredictionModel)
        assert isinstance(adapter, DataAdapter)

        # save it internally
        self.M = model

        # get the path count and train and target data
        [tra, targ] = adapter.get_complete_training_data()
        path_count = len(tra)
        permutation = np.random.permutation(path_count)

        # get the num
        num = int(np.ceil(path_count / N))

        # divide into validation and training data
        l = num * F
        r = num * (F + 1)

        # get slices
        slices_va = permutation[l:r]
        slices_tr = np.hstack([permutation[:l], permutation[r:]])

        # get filtered data
        filtered_va = [np.transpose(np.vstack([tra[slice] for slice in slices_va])), np.transpose(np.vstack([targ[slice] for slice in slices_va]))]
        filtered_tr = [np.transpose(np.vstack([tra[slice] for slice in slices_tr])), np.transpose(np.vstack([targ[slice] for slice in slices_tr]))]

        self.V = filtered_va
        self.T = filtered_tr

        progressbar_len = 40
        print(progressbar_len * "-")
        print("Validation set size: " + str(np.size(self.V[0], 1)))
        print("Train set size: " + str(np.size(self.T[0], 1)))

    # this method trains the internal prediction model
    def train(self, num_episodes, num_steps):

        progressbar_len = 40
        print(progressbar_len * "-")
        print("Training started:")

        # get some values
        [x, y] = self.T

        # for each episode
        eval_res = np.empty([2, num_episodes])

        # define progressbar length
        pbar = Progressbar(num_episodes, progressbar_len)

        # execute episodes
        for episode in range(num_episodes):

            # progress by one with the bar
            pbar.progress()

            # now we want to perform num_steps steps
            for num_step in range(num_steps):

                # simply perform a step with the model
                self.M.train(x, y)

            # save the evaluation result
            eval_res[0, episode] = self.validation_error()
            eval_res[1, episode] = self.train_error()

        print()
        print(progressbar_len * "-")

        return eval_res

    # this method evaluates the error on the validation set
    def validation_error(self):

        # this gets all validation data_adapter examples
        [x, y] = self.V

        # return the summed failure
        return 0.5 * np.mean(np.sum((self.M.predict(x) - y) ** 2, axis=0), axis=0)

    # this method evaluates the error on the validation set
    def train_error(self):

        # this gets all validation data_adapter examples
        [x, y] = self.T

        # return the summed failure
        return 0.5 * np.mean(np.sum((self.M.predict(x) - y) ** 2, axis=0), axis=0)
