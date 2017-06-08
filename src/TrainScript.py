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

# import the training controller
import matplotlib.pyplot as plt
from src.TrainingController import TrainingController
from src.PredictionModel import PredictionModel
from src.models.NeuralNetwork import NeuralNetwork

# this is the evaluation
trainroot = 'training_data/data_v0'
validationroot = 'validation_data/data_v0'
show_plots = True

# first of all create a  list of prediction models
models = list()

# add your models here
models.append(NeuralNetwork([90, 100, 100, 3]))

# foreeach model traijn the model and print the results7
for model in models:

    # check if model is prediction model
    assert isinstance(model, PredictionModel)

    # create a controller and traing it
    controller = TrainingController('training_data/data_v0', 'validation_data/data_v0', model)

    # get episode and step count from model and train consequently
    episodes = model.get_num_episodes()
    steps = model.get_num_steps()
    error = controller.train(episodes, steps)

    # # generate kind of a report
    report = "------------------------------\n" \
             + "Model: " + str(model.get_name()) + "\n" \
             + "Error-Rate: " + str(error[-1]) + "\n" \
             + "------------------------------"

    print(report)

    # if we want to show the plots
    if (show_plots):

        # now show the plot
        fig = plt.figure()

        # define title and
        plt.title(model.get_name())
        plt.plot(range(episodes), error)
        plt.show()