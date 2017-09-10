import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# This class holds and plots the trajectories and predictions
class HiddenTrajectoryPlot:
    def __init__(self, num_traj):
        """
        :param num_traj: The amount of training and validation trajectories that are plotted
        """
        self.fig = plt.figure(1)
        self.ax_arr = [None] * (num_traj * 2)

        # create all subplots
        for k in range(num_traj * 2):
            num = 2 * 100 + num_traj * 10 + (k + 1)
            self.ax_arr[k] = self.fig.add_subplot(num, projection='3d')

        self.trajectories_in = np.zeros((0, 0, num_traj))
        self.in_hidden = np.zeros((0, num_traj))
        self.in_velocity = np.zeros((0, 0, num_traj))
        self.in_direction = np.zeros((0, num_traj))
        self.pred_hidden = np.zeros((0, num_traj))
        self.pred_velocity = np.zeros((0, 0, num_traj))
        self.pred_direction = np.zeros((0, num_traj))
        self.fig.show()

    def update_trajectories(self,
                            trajectories_in: np.ndarray,
                            in_hidden: np.ndarray,
                            in_velocity: np.ndarray,
                            in_direction: np.ndarray,
                            pred_hidden: np.ndarray,
                            pred_velocity: np.ndarray,
                            pred_direction: np.ndarray) -> None:
        """
        Updates the trajectories that are plotted in the next call to plot.
        :param trajectories_in: The trajectories passed to the model
        """
        self.trajectories_in = trajectories_in
        self.in_hidden = in_hidden
        self.in_velocity = in_velocity
        self.in_direction = in_direction
        self.pred_hidden = pred_hidden
        self.pred_velocity = pred_velocity
        self.pred_direction = pred_direction

    def plot(self) -> None:
        """
        Plots the trajectories.
        """
        # print the trajectories
        for i in range(np.size(self.trajectories_in, 2)):
            self.ax_arr[i].cla()

            # print the input trajectory
            x = self.trajectories_in[0, :, i]
            y = self.trajectories_in[1, :, i]
            z = self.trajectories_in[2, :, i]
            self.ax_arr[i].plot(x, y, z, label='in')

            # print the predicted trajectory
            start_x = self.trajectories_in[0, -1, i]
            start_y = self.trajectories_in[1, -1, i]
            start_z = self.trajectories_in[2, -1, i]
            dx = self.in_velocity[0, 0, i]
            dy = self.in_velocity[1, 0, i]
            dz = self.in_velocity[2, 0, i]

            comb_x = start_x + dx
            comb_y = start_y + dy
            comb_z = start_z + dz
            self.ax_arr[i].plot([start_x, comb_x], [start_y, comb_y], [start_z, comb_z], label='real_vel',
                                color='#1455bc')

            rdx = self.pred_velocity[0, 0, i]
            rdy = self.pred_velocity[1, 0, i]
            rdz = self.pred_velocity[2, 0, i]

            rcomb_x = start_x + rdx
            rcomb_y = start_y + rdy
            rcomb_z = start_z + rdz
            self.ax_arr[i].plot([start_x, rcomb_x], [start_y, rcomb_y], [start_z, rcomb_z], label='pred_vel',
                                color='#1455bc')

            rz = self.in_hidden[0, i]
            self.ax_arr[i].scatter(0, 0, rz, label='real_hid', color='#d15555a0')

            # print the real trajectory
            z = self.pred_hidden[0, i]
            self.ax_arr[i].scatter(0, 0, z, label='pred_hid', color='#870404')

            self.ax_arr[i].title("Real {}; Pred {}".format(self.in_direction[0, i], self.pred_direction[0, i]))
            self.ax_arr[i].legend()


