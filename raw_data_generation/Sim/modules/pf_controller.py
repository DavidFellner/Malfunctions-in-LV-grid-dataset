import datetime
import os
import json
import time
import importlib

import pandas as pd

import pflib.pf as pf  # We do not need to mess around with the path - PFLib does this for us
import pflib.object_frame as pof

from experiment_config import experiment_path, chosen_experiment

spec = importlib.util.spec_from_file_location(chosen_experiment, experiment_path)
configuration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(configuration)


class PfController(object):
    def __init__(self, config: str = 'config.json'):
        if isinstance(config, str):
            try:
                with open(os.path.join(os.path.dirname(__file__), "..", config), "r") as f:
                    options = json.load(f)["pf_controller"]
            except Exception as e:
                raise e
        else:
            raise TypeError("Unsupported parameter type for config.")
        #self.grid_file = options["grid_file"]
        #self.pf_project_name = options["pf_project_name"]
        # self.hh_load_profiles_file = options["load_profiles_file"]
        #self.smartest_load_pf_name = options["smartest_load_name"]
        #self.smartest_node_pf_name = options["smartest_node_name"]
        # self.sim_time_params = options["simulation_time_params"]

        self.current_sim_time = None

        self.scen_name = None

        self.voltages_L1_df = None
        self.voltages_L2_df = None
        self.voltages_L3_df = None

        self.powers_L1_df = None
        self.powers_L2_df = None
        self.powers_L3_df = None

        self.malfunctioning_devices = None

        self.trafo_loading_df = None
        self.cable_loading_df = None

    def setup(self, **kwargs):
        #self.scen_name = kwargs["scen_name"]
        self.pf_project_name = kwargs["grid"]
        if not pf.isStarted:
            pf.start(inMemoryInstance="Data Generation")
        # pf.app.Show()
        try:
            self._activate_project()
        except pf.PFException:
            self._import_grid_file(kwargs["grid"])
            self._activate_project()
        self._prepare_loadflow()
        self._get_elements()

        self.current_sim_time = kwargs["start_sim_time"]
        total_sim_steps = kwargs["total_steps"]
        sim_step_delta_s = kwargs["sim_step_delta_s"]
        self.gridinfo = kwargs["current_grid_info"]
        self.malfunctioning_devices = self.gridinfo[4]

        sim_end_time = self.current_sim_time + total_sim_steps * datetime.timedelta(seconds=sim_step_delta_s)

        sim_time_range = pd.date_range(start=self.current_sim_time, end=sim_end_time, freq=f"{sim_step_delta_s}S")

        self._init_time_obj()

        x = self.nodes_pof.get_attributes("loc_name")
        #print(x, type(x))

        self.voltages_L1_df = pd.DataFrame(index=sim_time_range,
                                           columns=self.nodes_pof.index)
        self.voltages_L2_df = self.voltages_L1_df.copy()
        self.voltages_L3_df = self.voltages_L1_df.copy()

        self.powers_L1_df = pd.DataFrame(index=sim_time_range,
                                         columns=self.pf_all_loads_pof.index)
        self.powers_L2_df = self.powers_L1_df.copy()
        self.powers_L3_df = self.powers_L1_df.copy()

        self.trafo_loading_df = pd.DataFrame(index=sim_time_range, columns=self.pf_trafos_pof.index)
        self.cable_loading_df = pd.DataFrame(index=sim_time_range, columns=self.pf_cables_pof.index)

        pf.app.SetWriteCacheEnabled(1)

    def go(self, time_index, **kwargs):

        #print("pf_controller: perform go() with time_index = ", time_index)

        if "convergence_step" in kwargs:
            convergence_step = kwargs["convergence_step"]
        else:
            convergence_step = False

        self.current_sim_time = time_index

        self.time_obj.SetAttribute("cDate", self.current_sim_time.strftime("%Y%m%d"))
        self.time_obj.SetAttribute("cTime", self.current_sim_time.strftime("%H%M%S"))

        if "sim_comp_powers" in kwargs:
            #(pv_powers, bess_powers, hp_powers) = kwargs["sim_comp_powers"]
            (pv_powers, ev_powers) = kwargs["sim_comp_powers"]
            pv_powers /= 1000.  # convert from kW to MW
            #bess_powers /= 1000.
            #hp_powers /= 1000.
            ev_powers /= 1000.
            if configuration.QDSL_models_available:
                for pv in self.pf_pv_pof: pv.qdslCtrl.SetAttribute('e:initVals', list(pv_powers.loc[pv.loc_name].values))
                for ev in self.pf_ev_pof:
                    vals_list = ev.qdslCtrl.GetAttribute('e:initVals')
                    if ev in self.malfunctioning_devices:
                        vals_list = vals_list[2:4] + [1]
                    else:
                        vals_list = vals_list[2:4] + [0]
                    vals_list = [ev_powers.loc[ev.loc_name].values[0], ev_powers.loc[ev.loc_name].values[0] * 0.1875] + vals_list
                    ev.qdslCtrl.SetAttribute('e:initVals', vals_list)

            else:
                self.pf_pv_pof.set_attributes(pv_powers)        #ALSO SET Q!!! > check in PF
                #self.pf_bess_pof.set_attributes(bess_powers)
                #self.pf_hp_pof.set_attributes(hp_powers)
                self.pf_ev_pof.set_attributes(ev_powers)

        return_value = self.loadfl.Execute()
        if return_value != 0:
            #print("save current PF state - LF did not converge.")
            print("LF did not converge.")
            if "sim_comp_powers" in kwargs:
                err_dir = os.path.join(os.path.dirname(__file__), "..", "output", "error")
                if not os.path.isdir(err_dir):
                    os.makedirs(err_dir)

                #pv_powers.to_csv(os.path.join(err_dir, f"{self.scen_name}_err_pv.csv"))
                #hp_powers.to_csv(os.path.join(err_dir, f"{self.scen_name}_err_hp.csv"))
                #ev_powers.to_csv(os.path.join(err_dir, f"{self.scen_name}_err_ev.csv"))
                #bess_powers.to_csv(os.path.join(err_dir, f"{self.scen_name}_err_bess.csv"))

        # raise RuntimeError(f"Load flow calculation failed due to nonconvergence. Error Code: {return_value}")

        #voltages = self.nodes_pof.get_attributes(['m:u:A', 'm:u:B', 'm:u:C'])       # only m:u?
        voltages = self.nodes_pof.get_attributes(['m:u'])

        if not convergence_step or configuration.QDSL_models_available:

            '''self.voltages_L1_df.loc[self.current_sim_time] = voltages['m:u:A']  # .to_list() werden die richtigen werte in die richtige spalte geschrieben?
            self.voltages_L2_df.loc[self.current_sim_time] = voltages['m:u:B']  # .to_list()
            self.voltages_L3_df.loc[self.current_sim_time] = voltages['m:u:C']  # .to_list()'''

            self.voltages_L1_df.loc[self.current_sim_time] = voltages[
                'm:u']  # .to_list() #geht das so für symmetrisch?



            #powers = self.pf_all_loads_pof.get_attributes(["m:P:bus1:A", "m:P:bus1:B", "m:P:bus1:C"])
            powers = self.pf_all_loads_pof.get_attributes(["m:P:bus1"]) #geht das so für symmetrisch?
            powers *= 1000.  # convert from MW to kW

            '''self.powers_L1_df.loc[self.current_sim_time] = powers["m:P:bus1:A"]
            self.powers_L2_df.loc[self.current_sim_time] = powers["m:P:bus1:B"]
            self.powers_L3_df.loc[self.current_sim_time] = powers["m:P:bus1:C"]'''

            self.powers_L1_df.loc[self.current_sim_time] = powers["m:P:bus1"]

            self.trafo_loading_df.loc[self.current_sim_time] = self.pf_trafos_pof.get_attributes(["c:loading"])["c:loading"]  # double call needed because .get_attributes returns single column Dataframe but we need Series
            self.cable_loading_df.loc[self.current_sim_time] = self.pf_cables_pof.get_attributes(["c:loading"])["c:loading"]

        return (voltages, return_value)

        # if self.smartest_load_pf_name != "":
        #     smartest_voltage = voltages["m:u"][self.smartest_node_pf_name]
        #     return smartest_voltage

    def shutdown(self):
        pf.app.WriteChangesToDb()
        results = self._save_results()
        #pf.app.GetActiveProject().Deactivate()

        return results

    def _import_grid_file(self, file):
        #dir_path = os.path.dirname(__file__)
        #full_path = os.path.join(dir_path, "..", "input", self.grid_file)
        full_path = os.path.join(configuration.grid_data_folder, file)
        pf.pfd_import("\\USER", full_path + '.pfd')

    def _activate_project(self):
        err = pf.app.ActivateProject(self.pf_project_name)  # self.grid_file.split(".")[0]
        pf.PFException.raise_on_error(err, "Unable to activate the test project")

    def _get_elements(self):
        self.nodes_pof = pof.PFObjectFrame(pf.app.GetCalcRelevantObjects("*.ElmTerm"))
        #self.pf_pv_pof = pof.PFObjectFrame(pf.app.GetCalcRelevantObjects("PV_*.ElmLod"))
        self.pf_pv_pof = pof.PFObjectFrame([i for i in pf.app.GetCalcRelevantObjects("*.ElmGenStat") if i.outserv == 0])
        self.pf_ev_pof = pof.PFObjectFrame([i for i in pf.app.GetCalcRelevantObjects("* EVCS *.ElmLod") if i.outserv == 0])
        self.pf_bess_pof = pof.PFObjectFrame([i for i in pf.app.GetCalcRelevantObjects("BESS_*.ElmLod") if i.outserv == 0])
        self.pf_hp_pof = pof.PFObjectFrame([i for i in pf.app.GetCalcRelevantObjects("HP_*.ElmLod") if i.outserv == 0])
        self.pf_hh_loads_pof = pof.PFObjectFrame([i for i in pf.app.GetCalcRelevantObjects("* Load *.ElmLod") if i.outserv == 0])
        self.pf_all_loads_pof = pof.PFObjectFrame([i for i in pf.app.GetCalcRelevantObjects("*.ElmLod") if i.outserv == 0])

        self.pf_trafos_pof = pof.PFObjectFrame(pf.app.GetCalcRelevantObjects("*.ElmTr2"))
        self.pf_cables_pof = pof.PFObjectFrame(pf.app.GetCalcRelevantObjects("*.ElmLne"))

        self.lines_pof = pof.PFObjectFrame(pf.app.GetCalcRelevantObjects("*.ElmLne"))

    def _prepare_loadflow(self):
        #self.loadfl = pf.app.GetProjectFolder("study").SearchObject("Study Case.IntCase").SearchObject('*.ComLdf')
        self.loadfl = pf.app.GetActiveStudyCase().SearchObject('*.ComLdf')
        #self.loadfl.SetAttribute("iopt_net", 1)  # 0: AC Load Flow, balanced, positive sequence,
        # 1: AC Load Flow, unbalanced
        self.loadfl.SetAttribute("iopt_net", 0)  # 0: AC Load Flow, balanced, positive sequence,
        # 1: AC Load Flow, unbalanced
        self.loadfl.SetAttribute("iopt_pq", 1)  # Voltage dependency of loads

    def _init_time_obj(self):
        self.time_obj = pf.app.GetStudyTimeObject()
        self.time_obj.SetTime(0, 0)
        start_date = self.current_sim_time.strftime("%Y%m%d")
        start_time = self.current_sim_time.strftime("%H%M%S")
        self.time_obj.SetAttribute("cDate", start_date)
        self.time_obj.SetAttribute("cTime", start_time)
        # self.current_sim_time = datetime.datetime.strptime(start_date + start_time, "%Y%m%d%H%M%S")
        # self.step_delta = datetime.timedelta(hours=self.sim_time_params["sim_time_delta_h"])
        print("Start datetime set to", start_date, start_time)

    def _merge_voltages_dfs(self):
        self.voltages_L1_df.rename(columns={el: (el, "L1") for el in self.voltages_L1_df.columns}, inplace=True)
        self.voltages_L2_df.rename(columns={el: (el, "L2") for el in self.voltages_L2_df.columns}, inplace=True)
        self.voltages_L3_df.rename(columns={el: (el, "L3") for el in self.voltages_L3_df.columns}, inplace=True)
        df = pd.concat((self.voltages_L1_df, self.voltages_L2_df, self.voltages_L3_df), axis=1, sort=True)
        df = df.sort_index(axis=1)
        return df

    def _merge_power_dfs(self):
        self.powers_L1_df.rename(columns={el: (el, "L1") for el in self.powers_L1_df.columns}, inplace=True)
        self.powers_L2_df.rename(columns={el: (el, "L2") for el in self.powers_L2_df.columns}, inplace=True)
        self.powers_L3_df.rename(columns={el: (el, "L3") for el in self.powers_L3_df.columns}, inplace=True)
        df = pd.concat((self.powers_L1_df, self.powers_L2_df, self.powers_L3_df), axis=1, sort=True)
        df = df.sort_index(axis=1)
        return df

    def _save_results(self):
        merged_voltages_df = self._merge_voltages_dfs()
        merged_powers_df = self._merge_power_dfs()

        """out_dir = os.path.join(os.path.dirname(__file__), "..", "output", "scenarios", self.scen_name)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        print("Save results to Grid_Voltages.csv")
        merged_voltages_df.to_csv(os.path.join(out_dir, f"{self.scen_name}_Grid_Voltages.csv"))
        print("Save results to Sim_Powers.csv")
        merged_powers_df.to_csv(os.path.join(out_dir, f"{self.scen_name}_Sim_Powers.csv"))
        print("Save results to Loadings.csv")
        self.trafo_loading_df.to_csv(os.path.join(out_dir, f"{self.scen_name}_Trafo_Loadings.csv"))
        self.cable_loading_df.to_csv(os.path.join(out_dir, f"{self.scen_name}_Cable_Loadings.csv"))"""

        return (merged_voltages_df, merged_powers_df)


if __name__ == "__main__":
    my_pf_controller = PfController()
    my_pf_controller.setup(total_steps=10)
    start = time.time()
    for i in range(len(my_pf_controller.hh_profiles_df)):
        my_pf_controller.go(i)
    duration = time.time() - start
    print("pf_controller: Simulation took", duration, "s.")
    my_pf_controller.shutdown()
