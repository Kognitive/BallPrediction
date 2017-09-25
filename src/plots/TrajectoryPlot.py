import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# This class holds and plots the trajectories and predictions
class TrajectoryPlot:
    def __init__(self, num_traj):
        """
        :param num_traj: The amount of training and validation trajectories that are plotted
        """
        self.fig = plt.figure("Trajectories")
        self.ax_arr = [None] * (num_traj * 2)

        # create all subplots
        for k in range(num_traj * 2):
            num = 2 * 100 + num_traj * 10 + (k + 1)
            self.ax_arr[k] = self.fig.add_subplot(num, projection='3d')

        self.trajectories_in = np.zeros((0, 0, num_traj))
        self.trajectories_out = np.zeros((0, 0, num_traj))
        self.prediction_out = np.zeros((0, 0, num_traj))
        self.fig.show()

    def update_trajectories(self,
                            trajectories_in: np.ndarray,
                            trajectories_out: np.ndarray,
                            prediction_out: np.ndarray) -> None:
        """
        Updates the trajectories that are plotted in the next call to plot.
        :param trajectories_in: The trajectories passed to the model
        :param trajectories_out: The true continuation of the trajectories
        :param prediction_out: The predicted trajectories
        """
        self.trajectories_in = trajectories_in
        self.trajectories_out = trajectories_out
        self.prediction_out = prediction_out

    def plot(self) -> None:
        """
        Plots the trajectories.
        """
        # print the trajectories
        for i in range(np.size(self.trajectories_in, 2)):
            self.ax_arr[i].cla()

            # print the real trajectory
            x = self.trajectories_out[0, :, i]
            y = self.trajectories_out[1, :, i]
            z = self.trajectories_out[2, :, i]
            if len(x) == 1:
                self.ax_arr[i].scatter(x[0], y[0], z[0], label='Real')
            else:
                self.ax_arr[i].plot(x, y, z, label='Real')

            # print the predicted trajectory
            x = self.prediction_out[0, :, i]
            y = self.prediction_out[1, :, i]
            z = self.prediction_out[2, :, i]
            if len(x) == 1:
                self.ax_arr[i].scatter(x[0], y[0], z[0], label='Prediction')
            else:
                self.ax_arr[i].plot(x, y, z, label='Prediction')

            # print the input trajectory
            x = self.trajectories_in[0, :, i]
            y = self.trajectories_in[1, :, i]
            z = self.trajectories_in[2, :, i]
            self.ax_arr[i].plot(x, y, z, label='Input')
            self.ax_arr[i].legend()
