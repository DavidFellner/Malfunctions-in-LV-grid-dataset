
class BESS:

    def __init__(self, capacity, max_p, min_p, init_soc, init_power=0.0,
                 time_step_h=0.25, max_soc=1.0, min_soc=0.0, phase_id="plinir",
                 control_algorithm=None):

        self.capacity = capacity  # in kWh

        self.power = init_power  # in kW
        self.max_p = max_p  # in kW
        self.min_p = min_p  # in kW

        self.soc = init_soc  # in p.u.
        self.max_soc = max_soc  # in p.u.
        self.min_soc = min_soc  # in p.u.

        self.time_step_h = time_step_h  # in h

        self.phase_id = phase_id

        self.control_algorithm = control_algorithm

    def go(self, power, convergence_step=False):
        input = round(power, 4)
        new_soc = self.calc_soc(input)
        red_power, res_soc, red_required = self.check_soc_limits(new_soc)

        if red_required:
            self.power = self.check_p_limits(red_power)

        else:
            self.power = self.check_p_limits(input)

        if not convergence_step:
            self.soc = self.calc_soc(self.power)

        return self.power, red_required

    def check_soc_limits(self, new_soc):
        reduced_power = 0.0
        resulting_soc = 0.0
        reduction_required = False
        if new_soc >= self.max_soc:
            reduced_power = (self.max_soc - self.soc) * self.capacity / self.time_step_h
            resulting_soc = self.max_soc
            reduction_required = True
        elif new_soc <= self.min_soc:
            reduced_power = -(self.soc - self.min_soc) * self.capacity / self.time_step_h
            resulting_soc = self.min_soc
            reduction_required = True

        return reduced_power, resulting_soc, reduction_required

    def check_p_limits(self, power):
        if power >= self.max_p:
            return self.max_p
        elif power <= self.min_p:
            return self.min_p
        else:
            return power

    def calc_soc(self, power):
        if self.capacity != 0.0:
            curr_soc = self.soc
            return curr_soc + (power * self.time_step_h) / self.capacity
        else:
            return 0.0

