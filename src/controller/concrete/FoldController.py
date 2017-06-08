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
class FoldController(TrainingController):

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

        # fold the data
        data = adapter.get_complete_training_data()
        num = int(np.ceil(np.size(data[0], 1) / N))

        # divide into validation and training data
        l = num * F
        r = num * (F + 1)
        self.V = [data[0][:, l:r], data[1][:, l:r]]
        self.T = [np.hstack([data[0][:, :l], data[0][:, r:]]),
                  np.hstack([data[1][:, :l], data[1][:, r:]])]

    # this method trains the internal prediction model
    def train(self, num_episodes, num_steps):

        progressbar_len = 30
        print(progressbar_len * "-")
        print("Training started:")

        # get some values
        [x, y] = self.T

        # for each episode
        eval_res = np.empty([2, num_episodes])

        # define progressbar length
        batch_pack = num_episodes / progressbar_len
        it_batch_pack = batch_pack
        print(progressbar_len * '=')
        print(int(1 / batch_pack) * "=", end='', flush=True)

        for episode in range(num_episodes):

            # print just one character
            while (episode > int(it_batch_pack)):
                print('=', end='', flush=True)
                it_batch_pack = it_batch_pack + batch_pack

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
