import numpy as np
from scipy.signal import butter, lfilter
from src.data_adapter.DataAdapter import DataAdapter

# Applies a low pass filter to the data
class LowPassFilterAdapter(DataAdapter):

    def __init__(self, adapter: DataAdapter, cutoff = 1.2, fs = 120.0, order = 6):
        super().__init__()
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.raw_data = adapter.get_complete_training_data()
        self.data = [[self.__applyFilter(self.raw_data[i][j]) for j in range(len(self.raw_data[i]))] for i in range(len(self.raw_data))]
        # remove empty datasets
        self.data = [list(filter(lambda x: x.shape[0] != 0, self.data[i])) for i in range(len(self.data))]


    def get_complete_training_data(self):
        return self.data

    def __applyFilter(self, data):
        nyq = 0.5 * self.fs
        normal_cutoff = self.cutoff / nyq
        b, a = butter(self.order, normal_cutoff, btype='low', analog=False)
        filtered_data = np.zeros(data.shape)
        for i in range(data.shape[1]):
            filtered_data[:,i] = lfilter(b, a, data[:,i])
        # Cut off parts of the begin and of the end because low pass filters tend to produce unusable output at the ends
        return filtered_data[int(self.fs):len(filtered_data) - int(self.fs / 2),:]
