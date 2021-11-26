import json
import os

config = {
    "sim_manager":
        {
            "active_modules": (
                "pf_controller",
                "sim_comp_controller",
                "control_algorithm",
                "dut",
                "dewe_controller",
                # "pv_controller"
            ),
            "simulation_time_parameters": {
                "sim_start_datetime": "2018-03-03 00:00:00",  # format: "%Y-%m-%d %H:%M:%S"
                "sim_duration_s": 86400,  # 86400s = 24h - muss ganzzahliges vielfaches von 900 sein
                "sim_step_duration_s": 60.0,
            },
            "simulation_mode": "simulation",  # expected values are ["simulation", "emulation"]
            "voltage_convergence": False
        },
    "load_controller":
        {
            "load_profile": "rlc_load_profile.csv",
            "ipaddr": "192.168.1.71",
            "port": "502"
        },
    "dewe_controller":
        {
            "ipaddr": "192.168.1.10",
            "port": "502",
            "addresses": {"p_l1": 136,
                          "p_l2": 148,
                          "p_l3": 160,
                          "bat_u": 172}
        },
    "pf_controller":
        {
            "grid_file": "react_extended.pfd",
            "pf_project_name": "react_extended",
            "smartest_load_name": "smartest_lab",
            "smartest_node_name": "Urban LV B46(1)",
        },
    "dut_controller":
        {
            "broker_address": "147.91.50.80",
            "port": 8883,
            "username": "react_ait",
            "password": "8769n2v86co8976ov9746ov9voo",
            "device_id": "VICAIT123456789ABCDFE",  # "MID-00008484E913D3DB", "MEL-27597", "VICAIT123456789ABCDFE",
            "request_topic": "GWAIT_TEST_VIC1/request",  # "MID-DATAGATEWAYID001/request", # "MEL-GATEWAY-1/request",
            "response_topic": "GWAIT_TEST_VIC1/response",  # "MID-DATAGATEWAYID001/response", "MEL-GATEWAY-1/response",
            "data_topic": "GWAIT_TEST_VIC1/data",  # "MID-DATAGATEWAYID001/data", # "MEL-GATEWAY-1/data",
            "control_parameter": "pacGridSetPoint"  # "PacGridSetPoint", "SetTemperature"
        },
    "control_algorithm":
        {
            "algorithm_type": "schedule",  #  ["p_of_u", "schedule", "test_ramp", "constant", "self_consumption"(deprecated)]
            "dut_control_curve": "input/hil_net_demand_load_758_10_no_PV.csv",  # "input/hil_net_demand_load_758_10.csv",
            "sim_data_delta_s": 900
        },
    "sim_comp_controller":
        {
            "pv_profile": "pv_profile.csv",
            "bess_control": "self_consumption",
            "hp_control": "fixed_target",
            "hp_demand_file": "hp_demand.csv",
            "out_temp_file": "outdoor_temp.csv",
            "target_temp_file": "target_temperature.csv",
            "sim_data_delta_s": 900,
            "bess_parameters": {"capacity": 6.8,
                                "max_p": 2.0,
                                "min_p": -2.0}
        },
    "pv_controller":
        {
            "ipaddr": "192.168.1.9",
            "port": 5021,
            "pv_mult": 1.0,
            "pv_profile": "pv_profile.csv",
            "pv_data_delta_s": 900
        },
    "intelligent_pf_loads":
        [
            "lod_707_1",
            "lod_708_1",
            "lod_709_1",
            "lod_710_1",
            "lod_711_1",
            "lod_712_1",
            "lod_713_1",
            "lod_715_1",
            "lod_717_1",
            "lod_719_1",
            "lod_720_1",
            "lod_722_1",
            "lod_724_1",
            "lod_726_1",
            "lod_728_1",
            "lod_730_1",
            "lod_732_1",
            "lod_734_1",
            "lod_736_1",
            "lod_738_1",
            "lod_740_1",
            "lod_742_1",
            "lod_744_1",
            "lod_745_1",
            "lod_748_1",
            "lod_749_1",
            "lod_751_1",
            "lod_752_1",
            "lod_753_1",
            "lod_754_1",
            "lod_755_1",
            "lod_756_1",
            "lod_757_1",
            "lod_758_1",
            "lod_759_1",
            "lod_760_1",
            "lod_762_1",
            "lod_763_1",
            "lod_764_1",
            "lod_765_1",
            "lod_766_1",
            "lod_767_1",
            "lod_769_1",
            "lod_771_1",
            "lod_772_1",
            "lod_773_1",
            "lod_774_1",
            "lod_775_1",
            "lod_776_1",
            "lod_777_1",
            "lod_778_1",
            "lod_779_1",
            "lod_781_1",
            "lod_782_1",
            "lod_783_1",
            "lod_785_1",
            "lod_786_1",
            "lod_788_1",
            "lod_789_1",
            "lod_790_1",
            "lod_791_1",
            "lod_792_1",
            "lod_793_1",
            "lod_795_1",
            "lod_796_1",
            "lod_797_1",
            "lod_798_1",
            "lod_799_1",
            "lod_801_1",
            "lod_802_1",
            "lod_803_1",
            "lod_804_1",
            "lod_815_1",
            "lod_817_1",
            "lod_819_1",
            "lod_826_1",
            "lod_829_1",
            "lod_839_1",
            "lod_843_1",
            "lod_871_1",
            "lod_883_1",
            "lod_925_1",
            "lod_928_1",
            "lod_929_1",
            "lod_935_1",
            "lod_937_1",
            "lod_938_1",
            "lod_939_1",
            "lod_940_1",
            "lod_945_1",
            "lod_946_1",
            "lod_948_1",
            "lod_954_1",
            "lod_963_1"
        ]
}

with open(os.path.join(os.path.dirname(__file__), "..", 'config.json'), 'w+', encoding='utf-8') as f:
    json.dump(config, f, ensure_ascii=False, indent=4)
