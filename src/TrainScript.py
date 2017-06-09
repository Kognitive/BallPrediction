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
import numpy as np

from src.controller.concrete.PathFoldController import PathFoldController
from src.data_adapter.concrete.KOffsetAdapter import KOffsetAdapter
from src.models.concrete.NeuralNetwork import NeuralNetwork

# this is the evaluation
adapter = KOffsetAdapter(20, 1, 1, 'training_data/data_v1')
show_plots = True
N = 5

# settngs from data
in_size = adapter.get_exact_input_size()
out_size = adapter.get_exact_output_size()

# choose model
model = NeuralNetwork([in_size, out_size])
episodes = 10
steps = 10

# define the overall error
overall_error = np.zeros([2, episodes])

# set the range
for k in range(N):

    report = (40 * "-") + "\n" \
             + "[" + str(k + 1) + "/" + str(N) + "] Model: " + str(model.get_name())
    print(report)

    # create a controller and traing it
    model.reset()
    controller = PathFoldController(adapter, model, k, N)

    # get episode and step count from model and train consequentl
    error = controller.train(episodes, steps)
    overall_error = overall_error + error

    # # generate kind of a report
    report = "Validation Error: " + str(error[0, -1]) + "\n" \
             + "Train Error: " + str(error[1, -1]) + "\n" \
             + (40 * "-") + "\n"

    print(report)

# normalize error
overall_error = overall_error / N

# if we want to show the plots
if (show_plots):

    # now show the plot
    fig = plt.figure()

    # define title and
    plt.title(model.get_name())
    plt.plot(range(episodes), overall_error[0, :], label='Validation')
    plt.plot(range(episodes), overall_error[1, :], label='Train')
    plt.legend()
    plt.xlim([0, episodes - 1])
    plt.show()