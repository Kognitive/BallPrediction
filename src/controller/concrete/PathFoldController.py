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
from src.models.RecurrentPredictionModel import RecurrentPredictionModel
from src.data_loader.DataLoader import DataLoader
from src.data_transformer.concrete.FeedForwardDataTransformer import FeedForwardDataTransformer

# this class is a basic controller
from src.utils.Progressbar import Progressbar


class ProgressBar(object):
    pass


class PathFoldController(TrainingController):

    # this constructor creates a new data_loader iterator
    # and saves the passed prediction model
    #
    #   loader - A data loader, which is capable of supplying the data.
    #   model - choose a model wisely
    #   F - which fold should be selected
    #   N - how many folds are there overall
    #
    def __init__(self, loader, transformer, model, batch_size, F, N):

        # check whether the prediction model is an instance of the
        # correct interface
        assert isinstance(model, RecurrentPredictionModel)
        assert isinstance(loader, DataLoader)

        # save it internally
        self.M = model
        self.batch_size = batch_size

        # get the path count and train and target data
        trajectories = loader.load_complete_data()
        path_count = len(trajectories)

        # sample random permutation
        permutation = np.random.permutation(path_count)

        # get the num
        num = int(np.ceil(path_count / N))

        # divide into validation and training data
        l = num * F
        r = num * (F + 1)

        # get slices
        slices_va = permutation[l:r]
        slices_tr = np.hstack([permutation[:l], permutation[r:]])

        # divide into validation and training
        self.V = [trajectories[i] for i in slices_va]
        self.T = [trajectories[i] for i in slices_tr]

        print("Validation set size: " + str(len(self.V)) + " Trajectories")
        print("Train set size: " + str(len(self.T)) + " Trajectories")

        # transform the data
        self.transformer = transformer
        self.V = self.transformer.transform(self.V)
        self.T = self.transformer.transform(self.T)

        progressbar_len = 40
        print(progressbar_len * "-")

    # this method trains the internal prediction model
    def train(self, num_episodes, num_steps):

        progressbar_len = 40
        print(progressbar_len * "-")
        print("Training started:")

        # for each episode
        eval_res = np.empty([2, num_episodes])

        # define progressbar length
        pbar = Progressbar(num_episodes, progressbar_len)

        # execute episodes
        for episode in range(num_episodes):

            # progress by one with the bar
            pbar.progress()

            # sample the randomly
            slices = np.random.randint(0, np.size(self.T, 2), self.batch_size)

            # simply perform a step with the model
            self.M.train(self.T[:, :, slices], num_steps)

            # save the evaluation result
            eval_res[0, episode] = self.validation_error()
            eval_res[1, episode] = self.train_error()

        print()
        print(progressbar_len * "-")

        return eval_res

    # this method evaluates the error on the validation set
    def validation_error(self):

        overall_error = 0
        N = int(np.size(self.V, 2) / self.batch_size)

        # make packages
        for k in range(N):

            overall_error += self.M.validate(self.V[:, :, (k*self.batch_size):((k+1)*self.batch_size)])

        return overall_error

    # this method evaluates the error on the validation set
    def train_error(self):

        overall_error = 0
        N = int(np.size(self.V, 2) / self.batch_size)

        # make packages
        for k in range(N):
            overall_error += self.M.validate(self.T[:, :, (k * self.batch_size):((k + 1) * self.batch_size)])

        return overall_error
