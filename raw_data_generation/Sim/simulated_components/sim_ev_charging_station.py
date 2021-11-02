import os
import pandas as pd
from datetime import datetime
import importlib

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


class EVChargingStation:
    """
    Simple model for a EV charging station. The type is to be derived from teh profile used.

    :param yearly_demand - dataframe containing average heating demand for each hour of a year
    :param starting_temp - beginning temperature for simulation
    :param area - area of the building in square meters
    :param ceiling_height - height of ceiling throughout the building
    :param allowed_t_deviation - allowed deviation to stay within comfort zone

    """

    def __init__(self,
                 start_sim_time: datetime,
                 data_time_step_s: int = configuration.step_size * 60,
                 control_algorithm=None,
                 ev_chargingstation_type=None,
                 yearly_demand=None,
                 phase_id ="plinir",
                 size=None
                 ):

        self.ev_chargingstation_type = ev_chargingstation_type

        self.p_rated = float(ev_chargingstation_type.split('_')[2])

        self.data_time_step_s = data_time_step_s

        self.data_time_step = start_sim_time

        self.phase_id = phase_id

        self.control_algorithm = control_algorithm

        self.yearly_demand = yearly_demand

        self.p_lini = size


    def go(self, time_step, set_power, convergence_step=False):

        real_power = self.yearly_demand[time_step] * set_power * self.p_lini

        return [real_power]
