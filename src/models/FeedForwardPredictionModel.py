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

from src.data_transformer.concrete.FeedForwardDataTransformer import FeedForwardDataTransformer
from src.models.RecurrentPredictionModel import RecurrentPredictionModel
from src.utils.SetQueue import SetQueue

# This abstract class is a Feed Forward Prediction Model. It is a special
# case of a recurrent prediction model.


class FeedForwardPredictionModel(RecurrentPredictionModel):

    # This constructs the prediction model
    #
    # name - The name of the model
    # I - the number of position inputs
    # O - the number of position outputs
    # K - The offset for the prediction
    #
    def __init__(self, name, I, O, K):

        # pass the name and offset to the parent
        super().__init__(name, K)

        # create transformer with IOK
        self.transformer = FeedForwardDataTransformer(I, O, K)
        self.state_queue = SetQueue(3, I)

    # This method retrieves a list of trajectories. It can further
    # process or transform these trajectories. But the desired overall
    # behaviour is, that it should use the passed trajectories as training
    # data and thus adjust the parameters of the model appropriately.
    #
    # - trajectories: This is a list of trajectories (A trajectory is a numpy vector)
    #
    def train(self, trajectories, steps):

        # transform the trajectories
        [x, y] = self.transformer.transform(trajectories)
        self.fftrain(x, y, steps)

    # This method gets a the current state, and tries to output its prediction.
    #
    # - state This is basically one state, e.g. a x-y-z position
    #
    def validate(self, trajectories):

        # transform the trajectories
        [x, y] = self.transformer.transform(trajectories)

        # return the feed forward prediction
        return 0.5 * np.mean(np.sum((self.ffpredict(x) - y) ** 2, axis=0), axis=0)

    # This method has to be implemented, and the desired behaviour consists in resetting all
    # model internal parameters to
    def init_params(self):
        raise NotImplementedError("Please implement a init parameters method for " + str(self.name))

    # This method takes inputs and corresponding predictions and
    # appropriately trains the model internally.
    #
    # - inputs: The inputs for the model, based on which it should predicted.
    # - predictions: The predictions, such that an error can be calculated.
    # - steps: The number of steps the model should execute.
    #
    def fftrain(self, inputs, predictions, steps):
        raise NotImplementedError("Please implement a fftrain method for " + str(self.name))

    # this method should be used to predict the next state based
    # on the current inputs
    #
    # - input just the list of frames
    def ffpredict(self, input):
        raise NotImplementedError("Please implement a ffpredict method for " + str(self.name))
