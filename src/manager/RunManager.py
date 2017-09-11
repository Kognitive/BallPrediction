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

import os
import datetime
import configparser

from os.path import join
from localconfig import config
from src.manager.StatManager import StatManager


class RunManager:

    def __init__(self, conf: dict):
        """This class can be used to manage the access to the LogDirectory.

        Args:
            config - A dict containing the configuration
            root_dir - The root directory
            reload_dir - None if it should not be reloaded, Otherwise the directory to load
            use_wizard - True, if a wizard should ask for the name
        """

        self.config = conf
        self.timestamp = "{:%Y-%m-%d_%H-%M-%S}".format(datetime.datetime.now())

        # check whether a new name has to be obtained using a wizard
        if self.config['reload_dir'] is None:
            self.output_dir, self.reload = self.create_directory_name()

        else:

            # get the output dir name and check it exists, if not raise an error
            self.output_dir = join(self.config['root_dir'], self.config['reload_dir'])
            if not os.path.exists(self.output_dir):
                raise RuntimeError("The path {} can't be reloaded.".format(self.output_dir))

            self.reload = True

        # Create the structures if necessary
        print("Init structure in {}.".format(self.output_dir))
        self.create_folder_structure()

        # reload the state if necessary
        if self.reload:
            config.read(join(self.output_dir, '{}/state.ini'.format(self.config['reload_checkpoint'])))
            state = dict(config.items("State"))
            self.current_episode = state['current_episode']
            self.best_episode = state['best_episode']
            self.best_value = state['best_value']

        else:
            self.current_episode = 0
            self.best_episode = -1
            self.best_value = 0

    def inc_episode(self):
        self.current_episode += 1

    def save(self, checkpoint):

        # create the config
        parser = configparser.ConfigParser()
        parser.add_section("State")

        # add all configuration parameters
        parser.set("State", "current_episode", str(self.current_episode))
        parser.set("State", "best_episode", str(self.best_episode))
        parser.set("State", "best_value", str(self.best_value))

        # and write to disk
        with open(join(self.output_dir, '{}/state.ini'.format(checkpoint)), 'w') as cfg_file:
            parser.write(cfg_file)

        self.model.save(checkpoint)

    def init_model(self, model_details):

        [self.model_config, self.model] = model_details

        # When the model is not reloaded put in some details
        if not self.reload:
            self.model_config['log_dir'] = self.output_dir
            self.model_config['time_stamp'] = self.timestamp

        # Otherwise reload from the specified folder
        else:

            # reload the config
            config.read(join(self.output_dir, 'configuration.ini'))
            self.model_config = dict(config.items("Model"))


        self.model = self.model(self.model_config)
        self.model.init_params()
        self.model_config['model_params'] = self.model.num_params()

        print("-" * 80)
        self.write_model_config()

        if self.reload:
            self.model.restore(self.config['reload_checkpoint'])

    def create_stats(self):
        return StatManager(self.model_config['episodes'],
                           self.output_dir,
                           self.config['output_grouping'],
                           self.reload,
                           self.config['set_labels'])

    def create_folder_structure(self):
        """This method creates the folder structure if necessary."""

        if not self.reload:
            print("Creating folder structure")
            os.makedirs(join(self.output_dir, 'general'))
            os.makedirs(join(self.output_dir, 'best'))

    def write_model_config(self):
        """This method writes the passed model configuration inside of the output
        directory.
        """

        print("Saving model configuration to disk.")

        # create the config
        parser = configparser.ConfigParser()
        parser.add_section("Model")

        # add all configuration parameters
        for key in self.model_config.keys():
            parser.set("Model", key, str(self.model_config[key]))

        # and write to disk
        with open(join(self.output_dir, 'configuration.ini'), 'w') as cfg_file:
            parser.write(cfg_file)

    def create_directory_name(self):

        # extract important vars
        log_dir = self.config['root_dir']
        directory_exists = False

        if self.config['use_wizard']:
            print("Greetings!")
            print("I see you are here to start a new run or continue a previous run!")

            # Ask for a name until the user chose either a name that does not exist already
            #  or confirms he wants to continue the specified run
            while True:
                name = input("Please name the run: ")
                output_dir = '{}/{}'.format(log_dir, name)
                if os.path.exists(output_dir):
                    answer = input("I have found a run with this name. Would you like to continue [(Y)es|No]? ")
                    if answer == 'Yes' or answer == 'yes' or answer == 'Y' or answer == 'y':
                        directory_exists = True
                        break
                    else:
                        print("Then please choose another name!")
                else:
                    os.makedirs(output_dir)
                    break
        else:
            output_dir = '{}/{}'.format(log_dir, self.timestamp)
            os.makedirs(output_dir)

        return output_dir, directory_exists
