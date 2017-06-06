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
from src.DataAdapter import DataAdapter
from src.PredictionModel import PredictionModel

# this class represents the train network task, which used the
# predicition model.
class TrainingController:

    # this constructor creates a new data iterator
    # and saves the passed prediction model
    def __init__(self, train_root, val_root, prediction_model):

        # check whether the prediction model is an instance of the
        # correct interface
        assert isinstance(prediction_model, PredictionModel)

        # save it internally
        self.I = DataAdapter(train_root)
        self.V = DataAdapter(val_root)
        self.P = prediction_model

    # this method trains the internal prediction model
    def train(self, num_episodes, num_steps):

        # get some values
        bs = self.P.get_batch_size()
        ins = self.I.get_input_size()
        ous = self.I.get_output_size()

        # for each episode
        eval_res = np.empty(num_episodes)
        for episode in range(num_episodes):

            # obtain the training data
            x = np.empty([ins, bs])
            y = np.empty([ous, bs])
            for batch_element in range(bs):

                [i, o] = self.I.next()
                x[batch_element, :] = i
                y[batch_element, :] = o

            # now we want to perform num_steps steps
            for num_step in range(num_steps):

                # simply perform a step with the model
                self.P.train(x, y)

            # save the evaluation result
            eval_res[episode] = self.evaluate()


    # this method evaluates the error on the validation set
    def evaluate(self):

        # this gets all validation data examples
        [x, y] = self.V.get_all()

        # return the summed failure
        return np.sum((self.P.predict(x) - y) ** 2, axis=0)
