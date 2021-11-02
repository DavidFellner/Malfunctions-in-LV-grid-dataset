import os
import json
import time

import pandas as pd
import numpy as np


class ControlAlgorithm(object):
    def __init__(self, config: str = 'config.json'):
        if isinstance(config, str):
            try:
                with open(os.path.join(os.path.dirname(__file__), "..", config), "r") as f:
                    options = json.load(f)["control_algorithm"]
            except Exception as e:
                raise e
        else:
            raise TypeError("Unsupported parameter type for config.")

        self.algorithm = None

        self.control_schedule_loc = None
        try:
            self.control_schedule_loc = options["dut_control_curve"]
        except:
            print("control curve location not given in config.json")

        self.sim_comp = True

        self.ramp = None

        self.supply = None
        self.consumption = None

        self.schedule = None

        self.u_lim_low = None
        self.u_lim_high = None
        self.p_max = None
        self.p_min = None

        self.control_curve = None

        self.ramp_count = 0

        self.constant_value = None

        self.sim_comp_time_step = None
        self.sim_data_delta_s = options["sim_data_delta_s"]

        self.nominal_voltage = 230.0
        self.p_u_characteristic = pd.DataFrame(
            data = {"u_ratio": [ 0.95, 0.9699, .97, 1.0299, 1.03,  1.05],
                    "power"  : [-2100,  -500,   -0,      0,  500,  2400]})


    def setup(self, **kwargs):

        if "algorithm_type" in kwargs:
            self.algorithm = kwargs["algorithm_type"]
        #else:
           # print("ControlAlgorithm setup function called without providing algorithm type!"
                  #"Default algorithm type from config used.")

        if self.algorithm == "test_ramp":
            self.ramp = list(np.arange(0.05, 1, 0.05))

        elif self.algorithm == "self_consumption":
            self.supply = kwargs["supply"]
            self.consumption = kwargs["consumption"]

        elif self.algorithm == "schedule" and kwargs["comp"] == "sim":
            self.schedule = kwargs["schedule"]

        elif self.algorithm == "schedule" and kwargs["comp"] == "dut":
            self.sim_comp = False
            start_sim_time = kwargs["start_sim_time"]

            self.sim_comp_time_step = start_sim_time
            self.schedule = pd.read_csv(self.control_schedule_loc, index_col=0, parse_dates=True)

        elif self.algorithm == "constant":

            self.constant_value = kwargs["setpoint"]

        elif self.algorithm == "p_of_u":
            self.u_lim_high = kwargs["u_lim_high"]
            self.u_lim_low = kwargs["u_lim_low"]
            self.p_min = kwargs["p_min"]
            self.p_max = kwargs["p_max"]

        elif self.algorithm == "cosphi(P)":
            self.control_curve = kwargs["control_curve"]


    def go(self, time_index, **kwargs):

        if not self.sim_comp:
            if (time_index - self.sim_comp_time_step).seconds == self.sim_data_delta_s:
                self.sim_comp_time_step = time_index
            control_time_index = self.sim_comp_time_step
        else:
            control_time_index = time_index

        if self.algorithm == "test_ramp":
            return self.test_ramp(**kwargs)
        elif self.algorithm == "p_of_u" and "voltage" in kwargs:
            power = self.p_of_u(voltage=kwargs["voltage"])
            return power
        elif self.algorithm == "cosphi(P)":
            q = self.cosphi_P(p=kwargs['p'])
            return q
        elif self.algorithm == "schedule":
            power = self.schedule_alg(time_index=control_time_index)
            return power
        elif self.algorithm == "self_consumption":
            power = self.self_consumption(time_index=control_time_index, **kwargs)
            return power
        else:
            raise ValueError(f"Algorithm type not set {self.algorithm} or necessary inputs not provided for algorithm.")

    def shutdown(self):
        pass

    def cosphi_P(self, p):

        closest_value = self.control_curve.iloc[(self.control_curve['P'] - p).abs().argsort()[:1]]
        q = self.control_curve['Q'][closest_value.index].values[0]

        return q

    def p_of_u(self, voltage):
        power = 0.0

        if voltage <= self.u_lim_low:
            power = self.p_min
        elif voltage >= self.u_lim_high:
            power = self.p_max
        elif self.u_lim_low < voltage < self.u_lim_high:
            power = round(self.p_min + (self.p_max - self.p_min) * (voltage - self.u_lim_low) /
                          (self.u_lim_high - self.u_lim_low))

        return power  # negativ da auf den bezugspunkt vom netz geregelt wird

    def p_of_u_with_setpoint(self, voltage, external_setpoint=None):

        power = 0

        # converting voltage to p.u.
        voltage_ratio = voltage / self.nominal_voltage

        # filtering out 2 closest u_ratio points in the p_u characteristic
        pos = (self.p_u_characteristic["u_ratio"] - voltage_ratio).abs().argsort()[:2]
        closest_u = self.p_u_characteristic.iloc[pos].sort_values(by="u_ratio")

        # interpolation to obtain power set-point
        if voltage_ratio <= closest_u["u_ratio"][0]: # if smaller than all u_ratios
            power = closest_u["power"][0] # take lowest power from p_u_characteristic
        elif voltage_ratio >= closest_u["u_ratio"][1]: # if higher than all u_ratios
            power = closest_u["power"][1] # take highest power from p_u_characteristic
        else:
            slope = ( closest_u["power"][1] - closest_u["power"][0] ) / \
                    ( closest_u["u_ratio"][1] - closest_u["u_ratio"][0] )
            power = closest_u["power"][0]
            power += slope * (voltage_ratio - closest_u["u_ratio"][0])

        if external_setpoint is not None:
            # if p-u controller is in the "dead-band", then apply external setpoint
            if abs(power) < 1e-3:
                power = external_setpoint

        return power

    def p_of_u_fraunhofer(self, voltage):

        power = 0

        # converting voltage to p.u.
        voltage_ratio = voltage / self.nominal_voltage

        # filtering out 2 closest u_ratio points in the p_u characteristic
        pos = (self.p_u_characteristic["u_ratio"] - voltage_ratio).abs().argsort()[:2]
        closest_u = self.p_u_characteristic.iloc[pos].sort_values(by="u_ratio")

        # interpolation to obtain power set-point
        if voltage_ratio <= closest_u["u_ratio"][0]: # if smaller than all u_ratios
            power = closest_u["power"][0] # take lowest power from p_u_characteristic
        elif voltage_ratio >= closest_u["u_ratio"][1]: # if higher than all u_ratios
            power = closest_u["power"][1] # take highest power from p_u_characteristic
        else:
            slope = ( closest_u["power"][1] - closest_u["power"][0] ) / \
                    ( closest_u["u_ratio"][1] - closest_u["u_ratio"][0] )
            power = closest_u["power"][0]
            power += slope * (voltage_ratio - closest_u["u_ratio"][0])

        return power

    def self_consumption(self, time_index, consumption=None, supply=None):

        if consumption and supply:
            power = supply - consumption
        else:
            power = self.supply[time_index] - self.consumption[time_index]
        # negative power means charging of the battery - positive power means discharging
        return power

    def schedule_alg(self, time_index):
        if self.sim_comp:
            power = self.schedule[time_index]
        else:
            power = self.schedule.loc[time_index][0]
        # negative power means charging of the battery - positive power means discharging
        return power

    def test_ramp(self, **kwargs):
        ret_power = self.ramp[self.ramp_count % len(self.ramp)]
        self.ramp_count += 1
        return ret_power

if __name__ == '__main__':

    control_algorithm = ControlAlgorithm()

    control_algorithm.setup()

    while True:
        control_algorithm.go(1)
        time.sleep(0.5)



