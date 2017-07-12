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


from src.data_filter.concrete.LowPassFilter import LowPassFilter
from src.controller.concrete.PathFoldController import PathFoldController
from src.data_loader.concrete.SimDataLoader import SimDataLoader
from src.models.concrete.NeuralNetwork import NeuralNetwork
from src.models.concrete.LSTM import LSTM
from src.data_normalizer.concrete.FrameNormalizer import FrameNormalizer
from src.data_normalizer.concrete.IdentityNormalizer import IdentityNormalizer
from src.data_transformer.concrete.FeedForwardDataTransformer import FeedForwardDataTransformer

# specify the result dir
result_dir = 'run'
data_dir = 'sim_training_data/data_v1'

# set normalizer
normalizer = IdentityNormalizer()
# normalizer = FrameNormalizer([-5, 5], [-1, 1])

episodes = 100
batch_size = 1000
steps = 10

# define IOK
I = 20

# define the transformer
transformer = FeedForwardDataTransformer(I)

# define the model
# def __init__(self, I, H, C, N, K):
model = LSTM(3, 20, batch_size)
# model = NeuralNetwork([I * 3] + 3 * [100] + [O * 3], I, O, K)

# d
NF = 10
show_plots = True

# this is the evaluation
loader = SimDataLoader(data_dir)
loader.set_data_normalizer(normalizer)

# define the overall error
overall_error = np.zeros([2, episodes])

# set the range
for k in range(NF):

    report = (40 * "-") + "\n" \
             + "[" + str(k + 1) + "/" + str(NF) + "] Model: " + str(model.name)
    print(report)

    # create a controller and train it
    model.init_params()
    controller = PathFoldController(loader, transformer, model, batch_size, k, NF)

    # get episode and step count from model and train consequentl
    error = controller.train(episodes, steps)
    overall_error = overall_error + error

    # # generate kind of a report
    report = "Validation Error: " + str(error[0, -1]) + "\n" \
             + "Train Error: " + str(error[1, -1]) + "\n" \
             + (40 * "-") + "\n"

    print(report)

# normalize error
overall_error = overall_error / NF
overall_error = np.sqrt(2 * overall_error)

# un normalize the error
# overall_error = normalizer.un_normalize(overall_error)

# if we want to show the plots
if (show_plots):

    # now show the plot
    fig = plt.figure()

    # define title and
    plt.title(model.name)
    plt.plot(range(episodes), overall_error[0, :], label='Validation')
    plt.plot(range(episodes), overall_error[1, :], label='Train')
    plt.legend()
    plt.xlim([0, episodes - 1])
    plt.show()