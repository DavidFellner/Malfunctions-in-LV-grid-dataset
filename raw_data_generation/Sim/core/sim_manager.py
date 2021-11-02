import os
import json
import time
import datetime
import importlib
import pandas as pd
import random
import numpy as np

from ..modules import pf_controller
from ..modules import control_algorithm
from ..modules import sim_comp_controller

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


class SimManager(object):
    def __init__(self, grid=None, gridinfo=None, config: str = 'config.json'):
        if isinstance(config, str):
            try:
                with open(os.path.join(os.path.dirname(__file__), "..", config), "r") as f:
                    options = json.load(f)["sim_manager"]
            except Exception as e:
                raise e
        else:
            raise TypeError("Unsupported parameter type for config.")
        self.modules = options["active_modules"]

        self.sim_mode = options["simulation_mode"]

        self.grid = grid
        self.gridinfo = gridinfo

        # self.sim_time_parameters = options["simulation_time_parameters"]
        # self.sim_duration = self.sim_time_parameters["sim_duration_s"]
        # self.sim_step_duration_s = self.sim_time_parameters["sim_step_duration_s"]

        sim_duration_s = configuration.sim_length * 24 * 60 * 60
        sim_step_duration_s = configuration.step_size * 60

        if configuration.dev_mode:      #only do 2 steps wehn in development mode
            sim_duration_s = 8 * sim_step_duration_s

        self.sim_duration = sim_duration_s
        self.sim_step_duration_s = sim_step_duration_s

        self.sim_step_delta = datetime.timedelta(seconds=self.sim_step_duration_s)

        # start_sim_time = datetime.datetime.strptime(
        #    self.sim_time_parameters["sim_start_datetime"],
        #    "%Y-%m-%d %H:%M:%S")

        self.t_start = self.gridinfo[2]
        self.t_end = self.gridinfo[3]
        self.start_sim_time = datetime.datetime.strptime(
            self.t_start.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

        # self.end_sim_time = self.start_sim_time + datetime.timedelta(seconds=self.sim_duration)
        self.end_sim_time = datetime.datetime.strptime(
            self.t_end.strftime("%Y-%m-%d %H:%M:%S"), "%Y-%m-%d %H:%M:%S")

        if configuration.dev_mode:
            self.end_sim_time = self.start_sim_time + datetime.timedelta(seconds=sim_duration_s)

        self.current_sim_time = self.start_sim_time

        available_modules = {"pf_controller": pf_controller.PfController,
                             "control_algorithm": control_algorithm.ControlAlgorithm,
                             "sim_comp_controller": sim_comp_controller.SimCompController}

        self.active_modules = {}
        self.pf_voltage = None
        self.pf_voltage_last_step = None

        # self.voltage_convergence = options["voltage_convergence"]
        self.voltage_convergence = True

        # print("\n***************** Active Modules: *****************")
        for module in self.modules:
            # print(module)
            self.active_modules[module] = available_modules[module]()
        # print("***************************************************\n")

    def setup(self):  # scen_name="scenario1"):

        for name, module in self.active_modules.items():
            module.setup(total_steps=(self.sim_duration / self.sim_step_delta.seconds),
                         start_sim_time=self.start_sim_time,
                         sim_step_delta_s=self.sim_step_delta.seconds,
                         end_sim_time=self.end_sim_time,
                         simulation_mode=self.sim_mode,
                         grid=self.grid,
                         current_grid_info=self.gridinfo,
                         )

        if "pf_controller" in self.active_modules \
                and "sim_comp_controller" in self.active_modules \
                and self.voltage_convergence and self.sim_mode == "simulation":
            sim_comp_powers = self.active_modules["sim_comp_controller"].go(self.start_sim_time, convergence_step=True)
            self.pf_voltage = self.active_modules["pf_controller"].go(self.start_sim_time,
                                                                      sim_comp_powers=sim_comp_powers,
                                                                      convergence_step=True)

    def go(self, time_index):

        # elif self.sim_mode == "simulation":
        if self.sim_mode == "simulation":

            if "pf_controller" in self.active_modules \
                    and "sim_comp_controller" in self.active_modules \
                    and not self.voltage_convergence:
                sim_comp_powers = self.active_modules["sim_comp_controller"].go(time_index)
                self.pf_voltage = self.active_modules["pf_controller"].go(time_index, sim_comp_powers=sim_comp_powers)

            elif "pf_controller" in self.active_modules \
                    and "sim_comp_controller" in self.active_modules \
                    and self.voltage_convergence:  # set voltage convergence to TRUE?? > seems necessary

                conv_step = 0
                self.pf_voltage_last_step = self.pf_voltage

                sim_comp_powers = self.active_modules["sim_comp_controller"].go(time_index, voltages=self.pf_voltage,
                                                                                convergence_step=True)
                self.pf_voltage = self.active_modules["pf_controller"].go(time_index, sim_comp_powers=sim_comp_powers,
                                                                          convergence_step=True)

                # while (self.pf_voltage - self.pf_voltage_last_step).max(axis=1).max() > 0.001 or conv_step > 10: #? >10??
                while (self.pf_voltage - self.pf_voltage_last_step).max(
                        axis=1).max() > 0.001 or conv_step < 10:

                    self.pf_voltage_last_step = self.pf_voltage
                    sim_comp_powers = self.active_modules["sim_comp_controller"].go(time_index,
                                                                                    voltages=self.pf_voltage,
                                                                                    convergence_step=True)
                    self.pf_voltage = self.active_modules["pf_controller"].go(time_index,
                                                                              sim_comp_powers=sim_comp_powers,
                                                                              convergence_step=True)

                    conv_step += 1

                sim_comp_powers = self.active_modules["sim_comp_controller"].go(time_index,
                                                                                voltages=self.pf_voltage,
                                                                                convergence_step=False)
                self.pf_voltage = self.active_modules["pf_controller"].go(time_index,
                                                                          sim_comp_powers=sim_comp_powers,
                                                                          convergence_step=False)

            else:
                raise ValueError(
                    f"Not all required modules for simulation in active_modules! check input: {self.active_modules}")

        else:

            raise ValueError(f"Invalid sim mode! check input: {self.sim_mode}")

    def shutdown(self):

        for name, module in self.active_modules.items():
            if name == 'pf_controller':
                results = module.shutdown()
            else:
                module.shutdown()

        return results

    def run_simulation(self):

        counter = 0 #error counter to differentiate between timeshift error and other errors
        while (self.current_sim_time - self.start_sim_time).total_seconds() <= self.sim_duration:
            try:
                t_before = time.time()
                self.go(self.current_sim_time)
                t_after = time.time()

                '''if self.sim_mode == "emulation":
                ex_time = t_after - t_before
                if ex_time < self.sim_step_duration_s:
                    time.sleep(self.sim_step_duration_s - ex_time)
                else:
                    raise ValueError("Your simulation timestep took longer than", self.sim_step_duration_s, "seconds.")'''

                self.current_sim_time += self.sim_step_delta

            except KeyError:
                if counter < 5:
                    self.current_sim_time += self.sim_step_delta
                    counter += 1
                    #print('DST gap in profile bridged')
                else:
                    raise Exception




    def set_times(self, file):
        if not configuration.t_start and not configuration.t_end:  # default > simulation time inferred from available load/generation profile data
            t_start = pd.Timestamp(
                pd.read_csv(os.path.join(configuration.grid_data_folder, file, 'LoadProfile.csv'), sep=';',
                            index_col='time').index[0], tz='utc')
            t_end = pd.Timestamp(
                pd.read_csv(os.path.join(configuration.grid_data_folder, file, 'LoadProfile.csv'), sep=';',
                            index_col='time').index[-1], tz='utc')

            if configuration.sim_length < 365:  # randomly choose period of defined length during maximum simulation period to simulate
                time_delta = int((t_end - t_start) / np.timedelta64(1, 's'))  # simulation duration converted to seconds
                seconds = random.sample([*range(0, time_delta, 1)], 1)[
                    0]  # pick random point of the maximum simulation period
                t_off = pd.Timedelta(str(seconds) + 's')  # set offset

                sim_start = (t_start + t_off).replace(hour=0, minute=0, second=0)
                t_sim_length = pd.Timedelta(str(configuration.sim_length) + 'd')
                sim_end = sim_start + t_sim_length

                if sim_end < t_end:
                    t_start = sim_start
                    t_end = sim_end
                    return t_start, t_end

        else:
            t_start = configuration.t_start
            t_end = configuration.t_end

        return t_start, t_end


if __name__ == "__main__":
    sim_manager = SimManager()
    sim_manager.setup()
    try:
        sim_manager.run_simulation()
    except Exception as e:
        sim_manager.shutdown()
        raise e
    sim_manager.shutdown()
