import os
import json
import time
import datetime
import importlib
import pflib.pf as pf

import pandas as pd
import numpy as np

from ..tools import configure_intelligent_loads
from ..simulated_components import sim_heatpump
from ..simulated_components import sim_bess
from ..simulated_components import sim_ev_charging_station
from ..simulated_components import sim_pv_inverter
from ..modules.control_algorithm import ControlAlgorithm
from ..tools.calc_global_net_demand import get_summed_curve
from ..tools import prepare_grid_data

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


class SimCompController(object):
    def __init__(self, config: str = 'config.json'):
        if isinstance(config, str):
            try:
                with open(os.path.join(os.path.dirname(__file__), "..", config), "r") as f:
                    options = json.load(f)["sim_comp_controller"]
            except Exception as e:
                raise e
        else:
            raise TypeError("Unsupported parameter type for config.")

        self.scen_name = None
        self.grid_name = None
        self.gridinfo = None

        # self.pv_profile = options["pv_profile"]
        self.pv_profile_file = 'RESProfile.csv'  # dynamic through paras / simbench recognition
        self.bess_control = options["bess_control"]
        self.hp_control = options["hp_control"]
        # self.out_temp_file = options["out_temp_file"]
        # self.target_temp_file = options["target_temp_file"]
        # self.hp_demand_file = options["hp_demand_file"]
        self.load_profiles_file = 'LoadProfile.csv'  # encompasses for SIMBENCH regular loads, EVCs, HPs; dynamic through paras / simbench recognition

        self.bess_parameters = options["bess_parameters"]

        self.sim_comp_time_step = None
        self.sim_data_delta_s = options["sim_data_delta_s"]

        self.loads_df = None

        self.hp_power_df = None

        self.control_curve = None
        self.control_curve_bess = None
        self.control_curve_hp = None
        self.control_curve_ev = None

        self.sim_heatpumps = {}
        self.sim_pv = {}
        self.sim_bess = {}
        self.sim_ev = {}

        self.hp_power_df = None
        self.pv_power_df = None
        self.bess_power_df = None
        self.ev_power_df = None

        self.hp_temp_df = None
        self.bess_soc_df = None

    def setup(self, **kwargs):
        """
        :param kwargs: scen_name (str)
        """

        self.grid_name = kwargs["grid"]
        self.gridinfo = kwargs["current_grid_info"]
        active_PVs = self.gridinfo[0]
        active_EVCs = self.gridinfo[1]
        malfunctioning_devices = self.gridinfo[4]

        pv_profile = pd.read_csv(os.path.join(configuration.grid_data_folder, self.grid_name, self.pv_profile_file),
                                 index_col=0, parse_dates=True, delimiter=';')

        load_profiles = pd.read_csv(
            os.path.join(configuration.grid_data_folder, self.grid_name, self.load_profiles_file),
            index_col=0, parse_dates=True, delimiter=';')

        self.loads_df = configure_intelligent_loads.main(self.grid_name)

        # ev_demand = 1 #???  ADD

        pv_profile *= 1  # scales individual module size

        start_sim_time = kwargs["start_sim_time"]
        end_sim_time = kwargs["end_sim_time"]

        self.sim_step_delta_s = kwargs["sim_step_delta_s"]

        sim_time_range = pd.date_range(start=start_sim_time, end=end_sim_time, freq=f"{self.sim_step_delta_s}S")

        self.sim_comp_time_step = start_sim_time

        # print("initialize intelligent loads for", kwargs["scen_name"])
        # in_dir = os.path.join(os.path.dirname(__file__), "..", "input", "scenarios")
        # self.loads_df = pd.read_csv(os.path.join(in_dir, kwargs["scen_name"] + ".csv"))

        in_dir = 'load data path'
        # self.loads_df = configure_intelligent_loads.main(self.grid_name)

        # n_flexible_loads = self.loads_df["n_BESS"].sum() + self.loads_df["n_HP"].sum()
        n_flexible_loads = 'number of EV charging stations'

        # self.control_curve = get_summed_curve(self.loads_df)
        # self.control_curve = get_summed_curve(self.loads_df
        # self.control_curve_bess = self.control_curve.copy(deep=True)
        # self.control_curve_hp = self.control_curve.copy(deep=True)

        """if n_flexible_loads > 0:        #what for?? > check in function get_summed_curve?
            self.control_curve[:] *= 1 / n_flexible_loads
            self.control_curve_bess[:] *= 1 / self.loads_df["n_BESS"].sum()
            self.control_curve_hp[:] *= 1 / self.loads_df["n_HP"].sum()

        else:
            self.control_curve[:] *= 0
            self.control_curve_bess[:] *= 0
            self.control_curve_hp[:] *= 0"""

        # np.random.seed(12345)

        for ev in active_EVCs:
            ev_demand = load_profiles[pf.get_referenced_characteristics(ev, 'plini')[0].loc_name]
            ev_chargingstation_type = pf.get_referenced_characteristics(ev, 'plini')[
                0].loc_name
            sim_ev = sim_ev_charging_station.EVChargingStation(
                yearly_demand=ev_demand,
                ev_chargingstation_type=ev_chargingstation_type,
                size=ev.plini,
                start_sim_time=start_sim_time,
                control_algorithm=ControlAlgorithm(),
                phase_id="plini"            # if 3 phase calc insert sth like "plinir_plinis_plinit"
            )

            if ev not in malfunctioning_devices:
                sim_ev.control_algorithm.setup(algorithm_type="p_of_u",
                                               # write into some sort of parameter vector? also to have functioning one / broken one upackable to u_lim , p_min...
                                               u_lim_high=1.05,
                                               u_lim_low=0.95,
                                               p_max=sim_ev.p_rated,
                                               p_min=sim_ev.p_rated * 0.1875,
                                               comp="sim")
            else:
                sim_ev.control_algorithm.setup(algorithm_type="p_of_u",
                                               u_lim_high=1.0,
                                               u_lim_low=1,
                                               p_max=sim_ev.p_rated,
                                               p_min=sim_ev.p_rated,
                                               comp="sim")

            # self.sim_ev[f"EV_{ev.loc_name.split(' ')[0]}"] = sim_ev             #bus name is added
            self.sim_ev[f"{ev.loc_name}"] = sim_ev  # bus name is added

        for pv in active_PVs:
            # add control in module?
            """self.sim_pv[f"PV_{pv.loc_name.split('_')}"] = sim_pv.PV(profile=pv_profile[start_sim_time:end_sim_time],
                                                     size=(pv.pgini, pv.qgini),
                                                     phase_id="plinir") #if 3 phase calc insert sth like "plinir_plinis_plinit"
                                                     """

            control_curve = pd.DataFrame()
            if pv.GetAttribute('pQPcurve'):
                control_curve_name = 'cosphi(P)'
                control_curve['P'] = pv.GetAttribute('pQPcurve').GetAttribute('Ppu')  # assigned Control curve
                control_curve['Q'] = pv.GetAttribute('pQPcurve').GetAttribute('Qpu')

            pv.SetAttribute('av_mode', 'constv')  # Control changed so as setting reactive power is possible

            sim_pv = sim_pv_inverter.PV(profile=pv_profile,
                                        size=pv.pgini,
                                        control_algorithm=ControlAlgorithm(),
                                        phase_id="pgini_qgini")  # if 3 phase calc insert sth like "pginir_pginis_pginit"

            sim_pv.control_algorithm.setup(algorithm_type=control_curve_name,
                                           control_curve=control_curve,
                                           comp="sim")

            self.sim_pv[f"{pv.loc_name}"] = sim_pv

        # self.ev_power_df = pd.DataFrame(index=self.sim_ev.keys(), columns=["plinir", "plinis", "plinit"], data=0)
        # self.pv_power_df = pd.DataFrame(index=self.sim_pv.keys(), columns=["plinir", "plinis", "plinit"], data=0)

        # symmetric:
        self.ev_power_df = pd.DataFrame(index=self.sim_ev.keys(), columns=["plini"], data=0)
        self.pv_power_df = pd.DataFrame(index=self.sim_pv.keys(), columns=["pgini", "qgini"], data=0)

        '''
            for _, row in self.loads_df.iterrows(): #correct load table?
                load_nr = row["node"].split(" ")[0]    
                pf_load = row["PF_Load"]        #check which value to get
                if " " in pf_load:
                    load_nr = pf_load.split(" ")[1]
                else:
                    load_nr = pf_load.split("_")[1]
        
                sim_hp = sim_heatpump.BuildingModelWithHeatPump(
                    starting_temp=np.random.normal(21., .5),
                    yearly_demand=hp_demand.loc[start_sim_time:end_sim_time],
                    temperature_profile=out_temp[start_sim_time:end_sim_time],
                    heat_pump_type="PUHZ_SW50VKA",
                    start_sim_time=start_sim_time,
                    time_step_s=self.sim_step_delta_s,
                    hh_mult_factor=row["n_HP"],
                    phase_id=row["phase_config"],
                    control_algorithm=ControlAlgorithm()
                )
            


            sim_hp.control_algorithm.setup(algorithm_type="schedule",
                                           schedule=self.control_curve_hp.loc[start_sim_time:end_sim_time],
                                           comp="sim")
            

            self.sim_heatpumps[f"HP_{load_nr}"] = sim_hp

            
            
            mult_bess = row["n_BESS"]

            bess = sim_bess.BESS(capacity=self.bess_parameters["capacity"] * mult_bess,
                                 max_p=self.bess_parameters["max_p"] * mult_bess,
                                 min_p=self.bess_parameters["min_p"] * mult_bess,
                                 init_soc=0.0, control_algorithm=ControlAlgorithm(),
                                 phase_id=row["phase_config"],
                                 time_step_h=self.sim_step_delta_s/3600.)
            

            bess.control_algorithm.setup(algorithm_type="schedule",
                                         schedule=self.control_curve_bess.loc[start_sim_time:end_sim_time] * mult_bess,
                                         comp="sim")

            self.sim_bess[f"BESS_{load_nr}"] = bess
            
        '''
        # self.target_temp_file = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "input", self.target_temp_file))
        # braucht man denk ich nicht - raumtemperatur bleibt konstant ich denke das is einfacher

        # self.hp_temp_df = pd.DataFrame(index=sim_time_range, columns=self.sim_heatpumps.keys())
        # self.bess_soc_df = pd.DataFrame(index=sim_time_range, columns=self.sim_bess.keys())

        # self.hp_power_df = pd.DataFrame(index=self.sim_heatpumps.keys(), columns=["plinir", "plinis", "plinit"], data=0)
        # self.bess_power_df = pd.DataFrame(index=self.sim_bess.keys(), columns=["plinir", "plinis", "plinit"], data=0)

    def go(self, time_step, **kwargs):
        """
        :return:
            pv_powers, ev_powers, bess_powers, hp_powers:  DataFrame, has to have a column containing the names of the elements
            that should be set as index with index name 'loc_name'. The rest of the columns represent the attributes
            to be set. E.g. 'plini'
        """

        if (time_step - self.sim_comp_time_step).seconds == self.sim_data_delta_s:
            self.sim_comp_time_step = time_step

        if "convergence_step" in kwargs:
            convergence_step = kwargs["convergence_step"]
        else:
            convergence_step = False

        if "voltages" in kwargs:
            voltages = kwargs["voltages"]
        else:
            voltages = None

        for name, pv in self.sim_pv.items():
            split_phase_id = pv.phase_id.split("_")
            if pv.control_algorithm.algorithm == 'cosphi(P)':
                q = pv.control_algorithm.go(self.sim_comp_time_step,
                                            p=pv.profile.loc[self.sim_comp_time_step].values[0])
            elif voltages is not None:
                q = pv.control_algorithm.go(self.sim_comp_time_step, voltage=voltages[name])
            else:
                q = 0.0

            power = pv.go(time_step=self.sim_comp_time_step, voltages=voltages, q=q)
            if len(power) < 3:
                for i in list(range(len(power))):
                    self.pv_power_df.loc[name, split_phase_id[i]] = power[i]
            elif len(power) >= 3:
                for i in range(len(power)):
                    self.pv_power_df.loc[name, split_phase_id[i]] = power[i] / 3
                    self.pv_power_df.loc[name, split_phase_id[i]] = power[i] / 3
                    self.pv_power_df.loc[name, split_phase_id[i]] = power[i] / 3

        for name, ev in self.sim_ev.items():
            split_phase_id = ev.phase_id.split("_")
            if voltages is not None:
                target_power = ev.control_algorithm.go(self.sim_comp_time_step, voltage=voltages.loc[name.split('@ ')[1]].values[0]) / ev.p_rated
            else:
                target_power = 1

            power = ev.go(time_step=time_step, set_power=target_power, convergence_step=convergence_step)
            if len(power) < 3:
                for i in list(range(len(power))):
                    try:
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i]
                    except ValueError:
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i][0]
            elif len(power) >= 3:
                for i in list(range(len(power))):
                    try:
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i] / 3
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i] / 3
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i] / 3
                    except ValueError:
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i][0] / 3
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i][0] / 3
                        self.ev_power_df.loc[name, split_phase_id[i]] = power[i][0] / 3


        '''
        
        for name, bess in self.sim_bess.items():
            split_phase_id = bess.phase_id.split("_")
            target_power = bess.control_algorithm.go(self.sim_comp_time_step)
            if len(split_phase_id) == 1:
                power, reduced = bess.go(power=target_power, convergence_step=convergence_step)  # 0.0) #
                self.bess_power_df.loc[name, split_phase_id[0]] = power
            elif len(split_phase_id) == 3:
                power, reduced = bess.go(power=target_power, convergence_step=convergence_step)  # 0.0) #
                self.bess_power_df.loc[name, split_phase_id[0]] = power / 3
                self.bess_power_df.loc[name, split_phase_id[1]] = power / 3
                self.bess_power_df.loc[name, split_phase_id[2]] = power / 3

            self.bess_soc_df.loc[time_step, name] = bess.soc
            
        for name, hp in self.sim_heatpumps.items():
            split_phase_id = hp.phase_id.split("_")
            target_power = hp.control_algorithm.go(self.sim_comp_time_step)
            if len(split_phase_id) == 1:
                self.hp_power_df.loc[name, split_phase_id[0]] = hp.go(time_step=time_step, set_power=target_power,
                                                                      convergence_step=convergence_step)
            elif len(split_phase_id) == 3:
                power = hp.go(time_step=time_step, set_power=target_power, convergence_step=convergence_step)
                self.hp_power_df.loc[name, split_phase_id[0]] = power / 3
                self.hp_power_df.loc[name, split_phase_id[1]] = power / 3
                self.hp_power_df.loc[name, split_phase_id[2]] = power / 3

            self.hp_temp_df.loc[time_step, name] = hp.room_temperature
        
        '''

        return (self.pv_power_df, self.ev_power_df)  # self.bess_power_df, self.hp_power_df

    def shutdown(self):
        self._save_results()

    def _save_results(self):
        """print("Save hp results to HP_Temps.csv")  # change to ev?
        out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "scenarios", self.scen_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)"""

        # self.bess_soc_df.to_csv(os.path.join(out_dir, f"{self.scen_name}_BESS_SoC.csv")) #not needed?

        return


if __name__ == '__main__':

    control_algorithm = SimCompController()
    configure_intelligent_loads.main(os.listdir(configuration.grid_data_folder)[0])  # saves loads configuration to csv

    control_algorithm.setup(scen_name="scenario1")

    time_index = 0
    while True:
        control_algorithm.go(time_index)
        time.sleep(0.5)
        time_index += 1
