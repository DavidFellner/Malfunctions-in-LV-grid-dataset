
class PV:

    def __init__(self, profile, size=None, control_algorithm=None, phase_id="plinir"):

        self.profile = profile

        self.p_gini = size

        self.control_algorithm = control_algorithm

        self.phase_id = phase_id

    def go(self, time_step, voltages, q):

        #return self.profile.loc[time_step, "ac_power_kW"] * self.mult_hh
        real_power_value = self.profile.loc[time_step].values[0] * self.p_gini
        if self.control_algorithm.algorithm == 'cosphi(P)':
            reactive_power_value = q * real_power_value
        else:
            reactive_power_value = 0

        return [real_power_value, reactive_power_value]

