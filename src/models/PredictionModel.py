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

# this class represents one implementation of a prediction network.


class PredictionModel:

    # this method is capable of retrieving the name from the model.
    def get_name(self):
        raise NotImplementedError("Please define a name for this model.")

    # this method should be implemented to provide a train method
    #
    # - inputs: The inputs for the model, based on which it should
    #           predict.
    # - predictions: The predictions, such that an error can be
    #                calculated
    def train(self, inputs, predictions):
        raise NotImplementedError("Please Implement this method")

    # this method should be used to predict the next state based
    # on the current inputs
    #
    # - inputs just the list of frames
    def predict(self, inputs):
        raise NotImplementedError("Please implement this method")

    # this method defines the batch size for a model
    def get_batch_size(self):
        raise NotImplementedError("Specify a batch size")

    # this method can be used to retrieve the numer of episodes
    def get_num_episodes(self):
        raise NotImplementedError("You have to specify the number of episodes.")

    # this method returns the number of steps
    def get_num_steps(self):
        raise NotImplementedError("Specify number of steps per episode.")

    # basically define a data_adapter adapter
    def get_data_adapter(self, root):
        raise NotImplementedError("Specify a data_adapter adapter.")