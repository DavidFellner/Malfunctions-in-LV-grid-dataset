import os
import pandas as pd

from ..tools import configure_intelligent_loads
from ..core.sim_manager import SimManager

#scenarios_df = pd.read_excel(os.path.join("..", "scenarios.xlsx"))


def run_simulations(grid, gridinfo):

    #old
    #configure_intelligent_loads.main(grid)  # saves loads configuration to csv

    sim_manager = SimManager(grid, gridinfo)
    sim_manager.setup()
    try:
        sim_manager.run_simulation()
    except Exception as e:
        sim_manager.shutdown()
        raise e
    results = sim_manager.shutdown()

    return results

