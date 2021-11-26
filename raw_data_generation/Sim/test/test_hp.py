import unittest
import random
import os

import simulated_components.sim_heatpump as heatpump
import pandas as pd

N_TESTS = 1000


# class TestHeatPump(unittest.TestCase):

# @staticmethod
def create_random_test_hp():
    demand = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", 'input/hp_demand.csv'), index_col=0,
                         parse_dates=True)
    temp_profile = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", 'input/outdoor_temp.csv'), index_col=0,
                               parse_dates=True)

    return heatpump.BuildingModelWithHeatPump(temperature_profile=temp_profile, yearly_demand=demand,
                                              heat_pump_type="PUHZ_SW50VKA", time_step_s=900)


def test_yearly_sim():
    # for i in range(N_TESTS):
    test_hp = create_random_test_hp()

    test_end = random.randint(0, len(test_hp.yearly_demand.index))
    test_range = test_hp.yearly_demand.index[0:-5]  # [test_end-100:test_end]

    t_df = pd.DataFrame(index=test_range, columns=["indoor_temp", "outdoor_temp", "power"])

    for time_step in test_range:
        power = test_hp.go(time_step=time_step)
        t_df.loc[time_step, "indoor_temp"] = test_hp.room_temperature
        t_df.loc[time_step, "outdoor_temp"] = test_hp.heat_pump.outdoor_temperature
        t_df.loc[time_step, "power"] = power

    print(t_df)


if __name__ == '__main__':
    # unittest.main()

    test_yearly_sim()
