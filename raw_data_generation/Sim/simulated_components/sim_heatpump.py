import os
import pandas as pd
from datetime import datetime

from ..simulated_components.hp_types import *


class BuildingModelWithHeatPump:
    """
    Simple for calculation room temperature from input deviation.

    :param yearly_demand - dataframe containing average heating demand for each hour of a year
    :param starting_temp - beginning temperature for simulation
    :param area - area of the building in square meters
    :param ceiling_height - height of ceiling throughout the building
    :param allowed_t_deviation - allowed deviation to stay within comfort zone

    """

    def __init__(self,
                 yearly_demand: pd.DataFrame,
                 temperature_profile: pd.DataFrame,
                 heat_pump_type: str,
                 start_sim_time: datetime,
                 starting_temp: float = 21.0,
                 control_target: float = 21.0,
                 area: float = 150.0,
                 ceiling_height: float = 2.4,
                 allowed_t_deviation: float = 1.0,
                 time_step_s: int = 15,
                 data_time_step_s: int = 900,
                 hh_mult_factor: float = 1.0,
                 phase_id: str = "plinir",
                 control_algorithm=None,
                 real_hp: bool = False
                 ):
        self.yearly_demand = yearly_demand

        self._area = area
        self._ceiling_height = ceiling_height
        self._volume_air = self._area * self._ceiling_height  # m^3

        self.room_temperature = starting_temp
        self.outdoor_temperature_profile = temperature_profile
        self.target_temperature = control_target

        self.control_target_temperature = control_target

        self.allowed_comfort_deviation = allowed_t_deviation

        self.time_step_s = time_step_s
        self.data_time_step_s = data_time_step_s

        self.data_time_step = start_sim_time

        self.hh_mult_factor = hh_mult_factor

        self.phase_id = phase_id

        self.control_algorithm = control_algorithm

        hp_config = HP_TYPES[heat_pump_type]

        if hp_config["atw"]:
            power_map = hp_config["t_ext_cons_map"]
            cop_map = hp_config["t_ext_cop_map"]
        else:
            power_map = hp_config["t_diff_cons_map"]
            cop_map = hp_config["t_diff_cop_map"]

        self.heat_pump = HeatPump(t_diff_start=0.0,
                                  t_out_start=starting_temp,
                                  hp_type_str=heat_pump_type,
                                  power_map=power_map,
                                  cop_map=cop_map)

        self._heat_capacity_air = 1.012  # kJ/kgK
        self._density_air = 1.18681  # kg/m^3
        # Assuming an altitude of 194 metres above mean sea level (the worldwide median altitude of human habitation),
        # an indoor temperature of 23 °C, a dewpoint of 9 °C (40.85% relative humidity),
        # and 760 mm–Hg sea level–corrected barometric pressure (molar water vapor content = 1.16%).

        self._specific_building_heat_cap = self._volume_air * self._heat_capacity_air * self._density_air
        # kJ / K

    def go(self, time_step, set_power=None, convergence_step=False):

        if (time_step - self.data_time_step).seconds == self.data_time_step_s:
            self.data_time_step = time_step

        if set_power:

            target_temp = self.target_temperature
            ref_t_diff = self.heat_pump.tempround(self.heat_pump.temperature_difference)
            ref_power = self.heat_pump._power_map[ref_t_diff] * self.hh_mult_factor

            if set_power > 0:
                if set_power - ref_power > 0:
                    self.control_target_temperature = target_temp + self.allowed_comfort_deviation

            elif set_power < 0:
                if set_power + ref_power < 0:
                    self.control_target_temperature = target_temp - self.allowed_comfort_deviation

        self.heat_pump.update_temperatures(self.outdoor_temperature_profile.loc[self.data_time_step, "temp"],
                                           (self.target_temperature -
                                            self.room_temperature))

        if self.control_target_temperature - self.room_temperature >= self.allowed_comfort_deviation or \
                self.heat_pump.active:

            self.heat_pump.calc_power()
            power = self.heat_pump.current_power

            self.heat_pump.active = True

        else:

            power = 0.0

        electrical_energy_diff = self.calc_power_diff(power)
        q = self.heat_pump.calc_q(electrical_energy_diff)

        if not convergence_step:
            self.set_new_t(q)
        self.check_t_diff()

        self.reset_control_target()

        return power * self.hh_mult_factor

    def set_new_t(self, heating_energy_kJ: float):
        self.room_temperature += heating_energy_kJ / self._specific_building_heat_cap

    def calc_power_diff(self, power_kW: float) -> float:
        return power_kW * self.time_step_s - self.yearly_demand.loc[
            self.data_time_step, "power in kW"] * self.time_step_s

    def check_t_diff(self):
        if self.room_temperature - self.control_target_temperature > self.allowed_comfort_deviation:
            self.heat_pump.active = False

    def reset_control_target(self):
        self.control_target_temperature = self.target_temperature


class HeatPump:

    def __init__(self, t_diff_start, t_out_start, hp_type_str, power_map, cop_map, atw_comp=True):
        self.temperature_difference = t_diff_start
        self.outdoor_temperature = t_out_start

        self.active = False

        self._hp_type_str = hp_type_str

        self._atw_component = atw_comp

        self.current_power = 0.0

        self._power_map = power_map
        self._cop_map = cop_map

        # target temperature - room temperature = temperature difference
        # negative temperature difference signifies cooling operation
        # positive temperature difference signifies heating operation

    @staticmethod
    def tempround(x, prec=0, base=1.0):
        return round(base * round(float(x) / base), prec)

    def calc_power(self):
        if self._atw_component:
            ref_temp = self.tempround(self.outdoor_temperature)
            in_range = (21.0 > ref_temp > -21.0)
        else:
            ref_temp = self.tempround(self.temperature_difference)
            in_range = (11.0 > ref_temp > -11.0)

        if in_range:
            self.current_power = self._power_map[ref_temp]
        else:
            self.current_power = 0.0

    def update_temperatures(self, t_out, t_diff):
        self.outdoor_temperature = t_out
        self.temperature_difference = t_diff

    def calc_q(self, electrical_power):
        if self._atw_component:
            ref_temp = self.tempround(self.outdoor_temperature)
        else:
            ref_temp = self.tempround(self.temperature_difference)

        if 21.0 > ref_temp > -21.0:
            return electrical_power / self._cop_map[ref_temp]
        else:
            return 0.0


class BuildingModelLabHeatPump:
    """
    Simple for calculation room temperature from input deviation.

    :param yearly_demand - dataframe containing average heating demand for each hour of a year
    :param starting_temp - beginning temperature for simulation
    :param area - area of the building in square meters
    :param ceiling_height - height of ceiling throughout the building
    :param allowed_t_deviation - allowed deviation to stay within comfort zone

    """

    def __init__(self,
                 yearly_demand: pd.DataFrame,
                 temperature_profile: pd.DataFrame,
                 start_sim_time: datetime,
                 starting_temp: float = 21.0,
                 control_target: float = 21.0,
                 area: float = 75.0,
                 ceiling_height: float = 2.4,
                 allowed_t_deviation: float = 1.0,
                 time_step_s: int = 60,
                 data_time_step_s: int = 900,
                 hh_mult_factor: float = 1.0,
                 phase_id: str = "plinir",
                 heat_pump_type="MUZ_LN25VG"
                 ):
        self.yearly_demand = yearly_demand

        self.red_factor_size = 0.5

        self._area = area
        self._ceiling_height = ceiling_height
        self._volume_air = self._area * self._ceiling_height  # m^3

        self.room_temperature = starting_temp
        self.outdoor_temperature_profile = temperature_profile
        self.target_temperature = control_target

        self.control_target_temperature = control_target

        self.allowed_comfort_deviation = allowed_t_deviation

        self.time_step_s = time_step_s
        self.data_time_step_s = data_time_step_s

        self.data_time_step = start_sim_time

        self.hh_mult_factor = hh_mult_factor

        self.phase_id = phase_id

        self.lab_temperature = starting_temp
        self.lab_t_offset = 0.0
        self.calc_lab_t_offset()

        self.lab_setpoint = control_target - self.lab_t_offset

        self._heat_capacity_air = 1.012  # kJ/kgK
        self._density_air = 1.18681  # kg/m^3

        hp_config = HP_TYPES[heat_pump_type]

        self.cop_map = hp_config["t_diff_cop_map"]

        # Assuming an altitude of 194 metres above mean sea level (the worldwide median altitude of human habitation),
        # an indoor temperature of 23 °C, a dewpoint of 9 °C (40.85% relative humidity),
        # and 760 mm–Hg sea level–corrected barometric pressure (molar water vapor content = 1.16%).

        self._specific_building_heat_cap = self._volume_air * self._heat_capacity_air * self._density_air
        # kJ / K

    def calc_lab_t_offset(self):
        self.lab_t_offset = self.tempround(self.room_temperature - self.lab_temperature, prec=1, base=0.5)

    @staticmethod
    def tempround(x, prec=0, base=1.0):
        return round(base * round(float(x) / base), prec)

    def go(self, time_step, measured_power, set_power, lab_t):

        if (time_step - self.data_time_step).seconds == self.data_time_step_s:
            self.data_time_step = time_step

        self.lab_temperature = lab_t
        self.calc_lab_t_offset()

        if set_power:

            target_temp = self.target_temperature

            if set_power > 0:
                self.control_target_temperature = target_temp + self.allowed_comfort_deviation

            elif set_power < 0:
                self.control_target_temperature = target_temp - self.allowed_comfort_deviation

        electrical_energy_diff = self.calc_power_diff(measured_power)

        q = electrical_energy_diff * self.cop_map[self.tempround(
            self.outdoor_temperature_profile.loc[self.data_time_step, "temp"] - self.room_temperature)]

        self.set_new_t(q)

        # print(f"lab t offset: {self.lab_t_offset}")
        # print(f"control target: {self.control_target_temperature}")

        lab_hp_setpoint = self.control_target_temperature - self.lab_t_offset

        self.reset_control_target()

        return lab_hp_setpoint

    def set_new_t(self, heating_energy_kJ: float):
        self.room_temperature += heating_energy_kJ / self._specific_building_heat_cap

    def calc_power_diff(self, power_kW: float) -> float:
        return abs(power_kW * self.time_step_s) - self.yearly_demand.loc[
            self.data_time_step, "power in kW"] * self.time_step_s * self.red_factor_size

    def reset_control_target(self):
        self.control_target_temperature = self.target_temperature

