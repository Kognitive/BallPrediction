import matplotlib.pyplot as plt
import numpy as np


# This class collects and plots the error
class ModelStatistics:
    def __init__(self):
        self.train_error = None
        self.train_error_single = None
        self.validation_error = None
        self.validation_error_single = None
        self.best_train_episode = -1
        self.best_validation_episode = -1
        self.best_train_error = -1
        self.best_validation_error = -1
        self.plot_single = False

    def init(self, train_error_single: np.ndarray, validation_error_single: np.ndarray) -> None:
        """
        Inits the model statistics from single training and validation error.
        The overall error is calculated by taking the mean of the single variable errors
        :param train_error_single: The train error of each variable
        :param validation_error_single: The validation error of each variable
        """
        self.validation_error_single = validation_error_single
        self.validation_error = np.mean(self.validation_error_single, axis=0)
        self.train_error_single = train_error_single
        self.train_error = np.mean(self.train_error_single, axis=0)

        self.best_validation_episode = np.argmin(self.validation_error)
        self.best_validation_error = np.min(self.validation_error)
        self.best_train_episode = np.argmin(self.train_error)
        self.best_train_error = np.min(self.train_error)

    def update(self,
               train_error: np.ndarray,
               train_error_single: np.ndarray,
               validation_error: np.ndarray,
               validation_error_single: np.ndarray) -> None:
        """
        Updates the training and validation error
        :param train_error: The overall training error (mean)
        :param train_error_single: The training error for each variable
        :param validation_error: The overall validation error (mean)
        :param validation_error_single: The validation error for each variable
        :return: 
        """
        if self.train_error is None:
            self.train_error = np.array([train_error])
            self.train_error_single = np.array([train_error_single]).T
            self.validation_error = np.array([validation_error])
            self.validation_error_single = np.array([validation_error_single]).T

            self.best_train_episode = 0
            self.best_validation_episode = 0
            self.best_train_error = train_error
            self.best_validation_error = validation_error
        else:
            self.train_error = np.hstack((self.train_error, train_error))
            if train_error_single.shape[0] != self.train_error_single.shape[0]:
                self.train_error_single = np.zeros((self.train_error.shape[0], train_error_single.shape[0]))
            self.train_error_single = np.hstack((self.train_error_single, np.array([train_error_single]).T))

            self.validation_error = np.hstack((self.validation_error, validation_error))
            # These checks are for backwards compatibility because one script saved the single error and the other the
            # overall error. Now they always use the single error and reconstruct the overall error
            if validation_error_single.shape[0] != self.validation_error_single.shape[0]:
                self.validation_error_single = np.zeros((self.validation_error.shape[0], validation_error_single.shape[0]))
            self.validation_error_single = np.hstack((self.validation_error_single,
                                                      np.array([validation_error_single]).T))

            if train_error < self.best_train_error:
                self.best_train_error = train_error
                self.best_train_episode = np.size(self.train_error, 0) - 1
            if validation_error < self.best_validation_error:
                self.best_validation_error = validation_error
                self.best_validation_episode = np.size(self.validation_error, 0) - 1

    def get_best_episode(self) -> tuple:
        """
        :return: The best train and validation episode
        """
        return self.best_train_episode, self.best_validation_episode

    def plot(self):
        """
        Plots the overall training and validation error.
        :return: Nothing
        """
        plt.figure('Error')
        plt.clf()
        x = np.linspace(0, len(self.validation_error) - 1, len(self.validation_error))
        if self.plot_single:
            count = np.size(self.train_error_single, 0)
            rows = int(np.ceil(count / 3))
            for i in range(count):
                ax = plt.subplot(rows, 3, i + 1)
                ax.title.set_text('Error is: ' + str(self.validation_error_single[i, -1]))
                ax.plot(x, self.validation_error_single[i, :], color='r', label='Validation Error')
                ax.plot(x, self.train_error_single[i, :], color='b', label='Train Error')
                ax.legend()
        else:
            plt.title("Error is: " + str(self.validation_error[-1]))
            plt.plot(x, self.validation_error, color='r', label='Validation Error')
            plt.plot(x, self.train_error, color='b', label='Train Error')
            plt.legend()
