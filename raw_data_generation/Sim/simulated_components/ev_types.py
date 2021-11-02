def create_t_map_exp(tminus10, tminus8, tminus6, tminus4, tminus2, tminus1, tplus1, tplus2, tplus4, tplus6, tplus8,
                     tplus10):
    map = {
        -27: tminus10,
        -26: tminus10,
        -25: tminus10,
        -24: tminus10,
        -23: tminus10,
        -22: tminus10,
        -21: tminus10,
        -20: tminus10,
        -19: tminus10,
        -18: tminus10,
        -17: tminus10,
        -16: tminus10,
        -15: tminus10,
        -14: tminus10,
        -13: tminus10,
        -12: tminus10,
        -11: tminus10,
        -10: tminus10,
        -9: (tminus8 + tminus10) / 2,
        -8: tminus8,
        -7: (tminus8 + tminus6) / 2,
        -6: tminus6,
        -5: (tminus6 + tminus4) / 2,
        -4: tminus4,
        -3: (tminus2 + tminus4) / 2,
        -2: tminus2,
        -1: tminus1,
        0: 0,
        1: tplus1,
        2: tplus2,
        3: (tplus2 + tplus4) / 2,
        4: tplus4,
        5: (tplus6 + tplus4) / 2,
        6: tplus6,
        7: (tplus6 + tplus8) / 2,
        8: tplus8,
        9: (tplus8 + tplus10) / 2,
        10: tplus10
    }
    return map


def create_t_map_data(exttminus20, exttminus15, exttminus10, exttminus7, extt2, extt7, extt12, extt15, extt20):
    map = {
        -20: exttminus20,
        -19: (exttminus15 * 1 / 5 + exttminus20 * 4 / 5),
        -18: (exttminus15 * 2 / 5 + exttminus20 * 3 / 5),
        -17: (exttminus15 * 3 / 5 + exttminus20 * 2 / 5),
        -16: (exttminus15 * 4 / 5 + exttminus20 * 1 / 5),
        -15: exttminus15,
        -14: (exttminus15 * 4 / 5 + exttminus10 * 1 / 5),
        -13: (exttminus15 * 3 / 5 + exttminus10 * 2 / 5),
        -12: (exttminus15 * 2 / 5 + exttminus10 * 3 / 5),
        -11: (exttminus15 * 1 / 5 + exttminus10 * 4 / 5),
        -10: exttminus10,
        -9: (exttminus10 * 2 / 3 + exttminus7 * 1 / 3),
        -8: (exttminus10 * 1 / 3 + exttminus7 * 2 / 3),
        -7: exttminus7,
        -6: (exttminus7 * 8 / 9 + extt2 * 1 / 9),
        -5: (exttminus7 * 7 / 9 + extt2 * 2 / 9),
        -4: (exttminus7 * 6 / 9 + extt2 * 3 / 9),
        -3: (exttminus7 * 5 / 9 + extt2 * 4 / 9),
        -2: (exttminus7 * 4 / 9 + extt2 * 5 / 9),
        -1: (exttminus7 * 3 / 9 + extt2 * 6 / 9),
        0: (exttminus7 * 2 / 9 + extt2 * 7 / 9),
        1: (exttminus7 * 1 / 9 + extt2 * 8 / 9),
        2: extt2,
        3: (extt2 * 4 / 5 + extt7 * 1 / 5),
        4: (extt2 * 3 / 5 + extt7 * 2 / 5),
        5: (extt2 * 2 / 5 + extt7 * 3 / 5),
        6: (extt2 * 1 / 5 + extt7 * 4 / 5),
        7: extt7,
        8: (extt7 * 4 / 5 + extt12 * 1 / 5),
        9: (extt7 * 3 / 5 + extt12 * 2 / 5),
        10: (extt7 * 2 / 5 + extt12 * 3 / 5),
        11: (extt7 * 1 / 5 + extt12 * 4 / 5),
        12: extt12,
        13: (extt12 * 2 / 3 + extt15 * 1 / 3),
        14: (extt12 * 1 / 3 + extt15 * 2 / 3),
        15: extt15,
        16: (extt15 * 4 / 5 + extt20 * 1 / 5),
        17: (extt15 * 3 / 5 + extt20 * 2 / 5),
        18: (extt15 * 2 / 5 + extt20 * 3 / 5),
        19: (extt15 * 1 / 5 + extt20 * 4 / 5),
        20: extt20,
    }
    return map


def get_electric_cons(heat_cap_map, cop_map):
    elec_cons_map = {}
    for key in heat_cap_map.keys():
        elec_cons_map[key] = heat_cap_map[key] / cop_map[key]

    return elec_cons_map


MUZ_LN25VG = {
    "max_kWe": 1.5,
    "atw": 0,
    "t_diff_cons_map": create_t_map_exp(tminus10=0.65, tminus8=0.65, tminus6=0.63, tminus4=0.6, tminus2=0.35,
                                        tminus1=0.215,
                                        tplus1=1.05, tplus2=1.05,
                                        tplus4=1.05, tplus6=1.05, tplus8=1.1, tplus10=1.1)
    ,
    "t_diff_cop_map": create_t_map_exp(tminus10=4.5, tminus8=4, tminus6=4, tminus4=3.5, tminus2=3, tminus1=2.5,
                                       tplus1=2.5, tplus2=3, tplus4=3.5,
                                       tplus6=4, tplus8=4, tplus10=4.5)
}

PUHZ_SW50VKA = {
    "max_kWe": 2.99,
    "atw": 1,
    "t_ext_heat_cap_map": create_t_map_data(exttminus20=0.0,
                                            exttminus15=3.0,
                                            exttminus10=4.1,
                                            exttminus7=4.8,
                                            extt2=5.0,
                                            extt7=5.5,
                                            extt12=6.4,
                                            extt15=7.0,
                                            extt20=7.9),
    "t_ext_cop_map": create_t_map_data(exttminus20=1.0,
                                       exttminus15=1.48,
                                       exttminus10=1.87,
                                       exttminus7=2.10,
                                       extt2=2.47,
                                       extt7=3.32,
                                       extt12=3.89,
                                       extt15=4.23,
                                       extt20=4.8),
}

PUHZ_SW50VKA["t_ext_cons_map"] = get_electric_cons(PUHZ_SW50VKA["t_ext_heat_cap_map"], PUHZ_SW50VKA["t_ext_cop_map"])

PUHZ_W85VHA2 = {"max_kWe": 5.29,
                "atw": 1,
                "t_ext_heat_cap_map": create_t_map_data(exttminus20=4.9,
                                                        exttminus15=6.1,
                                                        exttminus10=7.3,
                                                        exttminus7=8.0,
                                                        extt2=8.5,
                                                        extt7=9.0,
                                                        extt12=9.2,
                                                        extt15=9.3,
                                                        extt20=9.5),
                "t_ext_cop_map": create_t_map_data(exttminus20=1.70,
                                                   exttminus15=1.95,
                                                   exttminus10=2.19,
                                                   exttminus7=2.34,
                                                   extt2=2.89,
                                                   extt7=3.72,
                                                   extt12=4.17,
                                                   extt15=4.44,
                                                   extt20=4.89)
                }

PUHZ_W85VHA2["t_ext_cons_map"] = get_electric_cons(PUHZ_W85VHA2["t_ext_heat_cap_map"], PUHZ_W85VHA2["t_ext_cop_map"])

PUHZ_W112VHA = {"max_kWe": 6.785,
                "atw": 1,
                "t_ext_heat_cap_map": create_t_map_data(exttminus20=6.8,
                                                        exttminus15=8.4,
                                                        exttminus10=9.9,
                                                        exttminus7=10.9,
                                                        extt2=11.2,
                                                        extt7=11.2,
                                                        extt12=11.2,
                                                        extt15=11.2,
                                                        extt20=11.2),
                "t_ext_cop_map": create_t_map_data(exttminus20=1.64,
                                                   exttminus15=1.93,
                                                   exttminus10=2.25,
                                                   exttminus7=2.33,
                                                   extt2=2.93,
                                                   extt7=3.94,
                                                   extt12=4.67,
                                                   extt15=5.04,
                                                   extt20=5.58)
                }

PUHZ_W112VHA["t_ext_cons_map"] = get_electric_cons(PUHZ_W112VHA["t_ext_heat_cap_map"], PUHZ_W112VHA["t_ext_cop_map"])


HP_TYPES = {"MUZ_LN25VG": MUZ_LN25VG,
            "PUHZ_SW50VKA": PUHZ_SW50VKA,
            "PUHZ_W85VHA2": PUHZ_W85VHA2,
            "PUHZ_W112VHA": PUHZ_W112VHA}

# exp data ata unit (measured at constant 22 deg
# {-6: 0.63,
#                 -5.5: 0.63,
#                 -5: 0.62,
#                 -4.5: 0.62,
#                 -4: 0.6,
#                 -3.5: 0.6,
#                 -3: 0.57,
#                 -2.5: 0.35,
#                 -2: 0.35,
#                 -1.5: 0.25,
#                 -1: 0.215,
#                 -0.5: 0.2,
#                 0: 0,
#                 0.5: 1.05,
#                 1: 1.1,
#                 1.5: 1.05,
#                 2: 1.05,
#                 2.5: 1.05,
#                 3: 1.05,
#                 3.5: 1.05,
#                 4: 1.05,
#                 4.5: 1.05,
#                 5: 1.05,
#                 5.5: 1.05,
#                 6: 1.05}
