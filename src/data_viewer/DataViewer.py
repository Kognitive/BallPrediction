import numpy as np
import matplotlib.pyplot as plt

# import necessary packages
from src.data_manager.concrete.RealDataLoader import RealDataLoader
from src.data_filter.concrete.LowPassFilter import LowPassFilter

# load the data
loader = RealDataLoader('training_data/data_v1')
unfiltered_data = loader.load_complete_data()

# filter the data
x = LowPassFilter()
filtered_data = x.filter(unfiltered_data)

# select one trajectory
ind = 5

# create the corresponding plots
fig, axes = plt.subplots(3)
plabels = ["x", "y", "z"]

# iterate over the rows
for k in range(3):
    row = axes[k]
    row.set_ylabel(plabels[k])
    row.set_xlabel("t")
    row.plot(unfiltered_data[ind][60:-120, k])
    row.plot(filtered_data[ind][:, k])

plt.show()