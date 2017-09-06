import multiprocessing
import time
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime


class LogDirectoryHelper:
    @staticmethod
    def create(log_dir: str, timestamp: str, use_timestamp: bool) -> tuple:
        """
        This method creates a log directory for the run.
        If the directory already exists the user may continue the run.
        :param log_dir: The log directory where the directory is created or looked for
        :param timestamp: The current timestamp
        :param use_timestamp: If true the timestamp is used as name, otherwise the user is asked to name the folder
        :return:
        """
        reload = False
        if not use_timestamp:
            print("Greetings!")
            print("I see you are here to start a new run or continue a previous run!")
            # Ask for a name until the user chose either a name that does not exist already
            #  or confirms he wants to continue the specified run
            while True:
                name = input("Please name the run: ")
                output_dir = log_dir + "/" + name + "_RUNNING/"
                if os.path.exists(output_dir):
                    answer = input("I have found a run with this name. Would you like to continue [(Y)es|No]? ")
                    if answer == 'Yes' or answer == 'yes' or answer == 'Y' or answer == 'y':
                        reload = True
                        break
                    else:
                        print("Then please choose another name!")
                else:
                    os.makedirs(output_dir)
                    break
        else:
            output_dir = log_dir + "/" + timestamp + "_RUNNING/"
            os.makedirs(output_dir)

        print("Created log directory")

        if not os.path.exists(output_dir + "general"):
            if reload: raise RuntimeError("This run can't be reloaded")
            os.makedirs(output_dir + "general")

        # create the model folders
        if not os.path.exists(output_dir + "mod_tr"):
            os.makedirs(output_dir + "mod_tr")

        if not os.path.exists(output_dir + "mod_va"):
            os.makedirs(output_dir + "mod_va")

        return output_dir, reload
