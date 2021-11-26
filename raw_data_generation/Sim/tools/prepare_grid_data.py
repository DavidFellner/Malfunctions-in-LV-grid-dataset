import numpy as np
import pandas as pd
import random

random.seed(666)

import re
import os
import collections
import importlib

import itertools
from collections import Iterable

# e_mobil_noe_folder_path=r'Q:\2030\03 Projekte\DG KU\2.01.30170.1.0 - e-mobil EV Simulationen\\'
# e_mobil_noe_scenario_material_folder_path=r'Q:\2030\03 Projekte\DG KU\2.01.30170.1.0 - e-mobil EV Simulationen\
# 05 Simulationen\scenario_material\\'
# emobil_noe_scenario_name = test_scen[0]
#
# # Load mapping file. Cumulative mapping file (not meant to be changed for scenarios)
# df_mapping = pd.read_excel(e_mobil_noe_folder_path +
#                            r'05 Simulationen\Netzdaten\\' +
#                            'Netzelemente_v3highpen.xlsx',
#                            sheet_name='nodes loads mapping',
#                            dtype={'ev_1': str, 'ev_2': str})
#
# df_load_map = pd.read_csv(e_mobil_noe_scenario_material_folder_path +
#                           'df_input_map_loads_3P_high_pen_v1.csv',
#                           sep=';', decimal=',', index_col=0, header=0)
#
# # Load controller settings
# df_cont_setting = pd.read_csv(e_mobil_noe_scenario_material_folder_path +
#                               'df_cont_set_' + emobil_noe_scenario_name + '.csv',
#                               sep=',', decimal='.', index_col=0)


class ScenarioPreparator:

    def __init__(self, grid, loads_df, configuration, max_pen=0.8):
        self.new_df_mapping = None
        self.new_df_load_map = None
        self.new_df_cont_setting = None
        self.evse_dist = None
        self.ev_scen_df = None
        self.heatpump_df = None
        print("----------------------------------------------------")
        self.grid = grid
        self.loads_df = loads_df
        self.configuration = configuration
        self.yearly_consumptions_of_profiles = None

        self.max_pen = max_pen
        print("starting preparing {} for EV simulations.".format(self.grid))
        self.create_new_df_mapping()
        self.calc_yearly_consumption_of_profiles()
        self.create_new_df_load_map()
        self.create_evse_distribution()
        self.calc_yearly_consumption_of_profiles()
        self.generate_ev_scenario(penetration=self.max_pen)

    def create_evse_distribution(self):
        # evse_dist_path = os.path.join(os.path.dirname(__file__), r"evse_distribution")
        # self.evse_dist = pd.read_csv(os.path.join(evse_dist_path, grid + '_evse_distribution.csv'))

        evse_list = []
        for i, el in enumerate(self.grid.loads_df.index.str.split(' ')):
            evse_list.append('EVSE_' + el[-1])

        # anzahl_wohneinheiten = []
        # for list in self.oscd_grid.loads_df['Alias Name 1'].str.split(';'):
        #     anzahl_wohneinheiten.append(list[0].split(' ')[-1])

        # self.evse_df = pd.DataFrame(data={'Port 1': self.oscd_grid.loads_df['Port 1'].to_list(),},
        #                             # 'Anzahl Wohneinheiten': anzahl_wohneinheiten},
        #                             index=new_index)  # columns=['Port 1', 'Anzahl Wohneinheiten']

        # df = pd.DataFrame()
        # df['node_name'] = self.new_df_cont_setting['node_name'].tolist()
        # df['UN'] = 0.4
        # df['object_name'] = self.new_df_cont_setting.index
        # df['object_type'] = 'evse'
        # df['object_number'] = [row.split('_')[1] for row in df['object_name']]
        # self.new_df_mapping = pd.concat([self.new_df_mapping, df])
        # self.new_df_mapping.sort_values(by=['node_name'], inplace=True)

        self.new_df_load_map = self.new_df_load_map.assign(evse=evse_list)

    def create_new_df_mapping(self):
        new_df_mapping = pd.DataFrame()
        object_names = list()
        object_numbers = list()
        for row in self.loads_df.iterrows():
            name = row[1]['id']
            object_names.append(name)
            number = row[0]
            object_numbers.append(number)
        new_df_mapping['node_name'] = [str(el).replace(',', '').replace(' ', '_') for el
                                       in self.loads_df['node'].tolist()]
        new_df_mapping['UN'] = 0.4
        new_df_mapping['object_name'] = pd.Series(object_names)
        new_df_mapping['object_type'] = 'load'
        new_df_mapping['object_number'] = pd.Series(object_numbers)
        # IMPORTANT: now it does not cover all nodes, only all loads + their node location
        # ToDo: Add EV loads!
        self.new_df_mapping = new_df_mapping

    def calc_yearly_consumption_of_profiles(self):
        # simbench
        local_data_folder = os.path.join(self.configuration.grid_data_folder, self.grid.split('.')[0])
        filepath = os.path.join(local_data_folder, "LoadProfile.csv")
        profiles_df = pd.read_csv(filepath, delimiter=';')
        yearly_consumptions_of_profiles = {}

        for profile in profiles_df:
            if profile == 'time':
                continue
            else:
                yearly_consumptions_of_profiles[profile] = sum(profiles_df[profile]) / 4    #15 minutes values > durch 4??

        self.yearly_consumptions_of_profiles = yearly_consumptions_of_profiles

    def create_new_df_load_map(self):
        # some households (loads) also have a car and pv. This mapping is created here
        new_df_load_map = pd.DataFrame(index=self.loads_df.index)

        # search the Alias Name 1 column for the needed information
        count_of_households_list = list()
        yearly_consumption_list_kWh = list()
        load_type_list = list()

        """for row in self.loads_df.iterrows():
            if float(gewerbe_cons_substring) != 0.0:
                yearly_consumption_list_kWh.append(float(gewerbe_cons_substring))
                load_type_list.append('G0')

            elif float(cons_substring) != 0.0:
                yearly_consumption_list_kWh.append(float(cons_substring))
                load_type_list.append('H0')

            else:
                yearly_consumption_list_kWh.append(count_of_households_list[-1] * 3500)
                load_type_list.append('H0')

            except:
                print("WARNING: yearly consumption not found for {} - probably no years given".format(row))
                yearly_consumption_list_kWh.append(count_of_households_list[-1] * 3500)
                load_type_list.append('X0')
        new_df_load_map['count_of_households'] = count_of_households_list
        new_df_load_map['load_consumption'] = yearly_consumption_list_kWh
        # Todo: check empty entries and how we can fix that

        new_df_load_map['load_type'] = load_type_list
        new_df_load_map['node_name'] = [str(el).replace(',', '').replace(' ', '_') for el
                                       in self.loads_df['node'].tolist()]
        new_df_load_map.set_index(new_df_load_map.index.str.replace(' ', '_'), inplace=True)
        self.new_df_load_map = new_df_load_map"""

    # def create_new_df_cont_setting(self):
    #     # self.generate_and_distribute_evs(0.8)
    #     # ToDo Finish

    # def fill_new_df_mapping_with_evs(self):
    #     df = pd.DataFrame()
    #     df['node_name'] = self.new_df_cont_setting['node_name'].tolist()
    #     # df['UN'] = 0.4
    #     df['object_name'] = self.new_df_cont_setting.index
    #     df['object_type'] = 'ev'
    #     df['object_number'] = [row.split('_')[1] for row in df['object_name']]
    #     self.new_df_mapping = pd.concat([self.new_df_mapping, df], sort=True)
    #     self.new_df_mapping.sort_values(by=['node_name'], inplace=True)


    def generate_ev_scenario(self, penetration: float):         #!!!

        ev_name_type_list = self.generate_ev_list(penetration=penetration)

        ev_scen_df = pd.DataFrame(columns=["evse_home", "evse_shop", "evse_work"], index=ev_name_type_list, data=None)

        h0_evse_df = self.new_df_load_map.loc[self.new_df_load_map["load_type"] == "H0"]

        g0_evse_df = self.new_df_load_map.loc[self.new_df_load_map["load_type"] == "G0"]

        home_evse_list = self.generate_evse_list(h0_evse_df, ev_name_type_list, type="H0", mult=1)

        shop_work_evse_list = self.generate_evse_list(g0_evse_df, ev_name_type_list, type="G0", mult=4)

        home_i, shop_work_i = (0, 0)

        for name, row in ev_scen_df.iterrows():

            ev_scen_df.at[name, "evse_home"] = random.sample(home_evse_list, k=1)[0]
            home_i += 1
            ev_scen_df.at[name, "evse_shop"] = random.sample(shop_work_evse_list, k=1)[0]
            shop_work_i += 1
            ev_scen_df.at[name, "evse_work"] = random.sample(shop_work_evse_list, k=1)[0]
            shop_work_i += 1

        ev_scen_df.index.names = ["ev_name"]

        self.ev_scen_df = ev_scen_df

    def generate_evse_list(self, df, ev_list, type="H0", mult=1):

        weighted_list = [mult * row["count_of_households"] * [row['evse']] for i, row in df.iterrows()]

        flat_weighted_list = [x for sublist in weighted_list for x in sublist]

        if len(ev_list) > len(flat_weighted_list):

            add_list = []

            while len(add_list) < len(ev_list) - len(flat_weighted_list):

                if type == "H0":
                    add_list.append(np.random.choice(flat_weighted_list))
                else:
                    add_list.append("na")

            flat_weighted_list += add_list

        if type == "G0":

            flat_weighted_list += flat_weighted_list

        random.shuffle(flat_weighted_list)

        return flat_weighted_list

        # for name, row in ev_scen_df.iterrows():

    def generate_ev_list(self, penetration: float):

        total_households = self.new_df_load_map.loc[self.new_df_load_map["load_type"] == "H0"][
            "count_of_households"].sum()

        if 'Ländliches' in self.grid.file_name:
            count_evs = int(1.5 * total_households * penetration)  # number of EVs in the grid
            # ToDo: adapt to Austrian average values. count_evs defines the maximum number of EVs possible in a scenario
        elif 'Sub-Urbanes' in self.grid.file_name:
            count_evs = int(1.35 * total_households * penetration)
        elif ('Urbanes' in self.grid.file_name and
              'Sub' not in self.grid.file_name) or \
                ('Stätisches' in self.grid.file_name):
            count_evs = int(0.75 * total_households * penetration)
        else:
            raise ValueError('No correct Grid Name found')

        # self.new_df_load_map['ev_name'] = ''

        car_type_penetration = {'egolf': 0.325,
                                'zoe': 0.216,
                                'i3': 0.155,
                                'leaf': 0.118,
                                'ioniq': 0.094,
                                'models': 0.092}

        ev_list = ['ev_' + str(i + 1) for i in range(int(round(count_evs)))]

        ev_type_list = np.random.choice(list(car_type_penetration.keys()), len(ev_list),
                                        p=list(car_type_penetration.values()),
                                        replace=True)

        new_ev_list = []
        for i in range(len(ev_list)):
            new_ev_list.append(ev_list[i] + '_' + ev_type_list[i])

        return new_ev_list

    def generate_and_distribute_evs(self, penetration: float):
        # penetration can be element of [0.2, 0.5, 0.8]
        # this function also modifies self.new_df_load_map in place.
        # generate correct number of EVs:
        # total_households = self.new_df_load_map['count_of_households'].sum()

        new_ev_list = self.generate_ev_list(penetration=penetration)

        self.new_df_cont_setting = pd.DataFrame(index=new_ev_list, columns=['node_name'])
        # randomly distribute the elements of ev_list to the self.new_df_load_map['ev_name'] pandas Series
        # (multiple EVs per cell possible):             #done in reasonable manner(peak hours, weekdays?)
        # idx_list = list(range(len(self.new_df_load_map['ev_name']))

        idx_list = [row["count_of_households"] * [row['node_name']] for i, row in self.new_df_load_map.iterrows()]
        idx_list = list(itertools.chain.from_iterable(idx_list))

        random.shuffle(idx_list)
        index = 0
        while index < len(new_ev_list):
            while idx_list:
                try:
                    i = idx_list.pop()
                    # self.new_df_cont_setting.loc[new_ev_list[index], 'node_name'] =
                    # self.new_df_load_map['node_name'][i]
                    if self.new_df_load_map.iat[i, self.new_df_load_map.columns.get_loc('ev_name')] == '':
                        self.new_df_load_map.iat[i, self.new_df_load_map.columns.get_loc('ev_name')] = new_ev_list[
                            index]
                    else:
                        self.new_df_load_map.iat[i, self.new_df_load_map.columns.get_loc('ev_name')] = \
                            self.new_df_load_map.iat[i, self.new_df_load_map.columns.get_loc('ev_name')] \
                            + ';' + new_ev_list[index]
                    index += 1
                except IndexError:  # happens if index exceeds len(new_ev_list)
                    break
            idx_list = list(range(len(self.new_df_load_map['ev_name'])))
            random.shuffle(idx_list)

    def adapt_ev_profiles(self, penetration):

        if penetration < self.max_pen:
            print("Penetration is smaller that maximum penetration, start creation of corresponding ev scenario.")
        else:
            raise ValueError("Penetration is not smaller than max penetration! Cannot continue!")

        ratio = penetration / self.max_pen

        return self.ev_scen_df.sample(frac=ratio, random_state=42)

