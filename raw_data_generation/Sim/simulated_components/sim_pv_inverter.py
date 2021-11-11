import importlib

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)

class PV:

    def __init__(self, profile, size=None, control_algorithm=None, control_curve=None, phase_id="plinir"):

        self.profile = profile

        self.p_gini = size

        self.control_algorithm = control_algorithm

        self.control_curve = control_curve

        self.phase_id = phase_id

    def go(self, time_step, voltages=None, q=None):

        #return self.profile.loc[time_step, "ac_power_kW"] * self.mult_hh

        if configuration.QDSL_models_available:
            p = self.profile.loc[time_step].values[0]
            real_power_value = p * self.p_gini                  #p_gini set to 0 in pf to allow for proper function of QDSL model

            closest_value = self.control_curve.iloc[(self.control_curve['P'] - p).abs().argsort()[:1]]
            q = self.control_curve['Q'][closest_value.index].values[0]

            reactive_power_value = q * real_power_value

        else:
            real_power_value = self.profile.loc[time_step].values[0] * self.p_gini


            if self.control_algorithm.algorithm == 'cosphi(P)':
                reactive_power_value = q * real_power_value
            else:
                reactive_power_value = 0

        return [real_power_value, reactive_power_value]

