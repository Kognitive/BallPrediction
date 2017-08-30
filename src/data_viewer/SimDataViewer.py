import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import necessary packages
from src.data_loader.concrete.SimDataLoader import SimDataLoader
from src.data_transformer.FeedForwardDataTransformer import FeedForwardDataTransformer

# select one trajectory
ind = 126

# load the data
loader = SimDataLoader('sim_training_data/data_v2')
data = loader.load_single_datum(ind)

set_in, set_out = FeedForwardDataTransformer.transform([data], 10, 25)

# create the corresponding plots
fig, axes = plt.subplots(3)
plabels = ["x", "y", "z"]

# iterate over the rows
for k in range(3):
    row = axes[k]
    row.set_ylabel(plabels[k])
    row.set_xlabel("t")
    row.plot(data[:, k])


fig = plt.figure(str(ind))

# Plot the whole trajectory in 3d
ax = fig.add_subplot(121, projection='3d')
ax.plot(data[:, 0], data[:, 1], data[:, 2])

# Plot the sub-trajectories of the FeedForwardDataTransformer (going through each at 0.25s per trajectory)
ax = fig.add_subplot(122, projection='3d')
for i in range(0, np.size(set_in, 2)):
    ax.cla()

    x = set_in[0, :, i]
    y = set_in[1, :, i]
    z = set_in[2, :, i]
    ax.plot(x, y, z, label='In')

    x = set_out[0, :, i]
    y = set_out[1, :, i]
    z = set_out[2, :, i]
    ax.plot(x, y, z, label='Out')
    plt.pause(0.25)

plt.show()