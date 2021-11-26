import pandas as pd
import json
from datetime import datetime

import hpgen

fn = "input/Dublin weather data 2018.json"

io = open(fn, "r")

wd = json.load(io)

df = pd.DataFrame(columns=["month", "day", "hour", "temp"])

for day in wd["weather_data"]:

    hours = day["hour"]

    for hour in hours:
        date_time_obj = datetime.strptime(hour["time"], '%Y-%m-%d %H:%M')
        t = hour["temp_c"]
        df = df.append({"month": int(date_time_obj.month),
                        "day": int(date_time_obj.day),
                        "hour": int(date_time_obj.hour) + 1,
                        "temp": t},
                       ignore_index=True)

for colstr in ["month", "day", "hour"]:
    df[colstr] = df[colstr].astype('int')

temperature_timeseries, hp_demand = hpgen.create_hp_profiles(weather_data=df,
                                                             month_id="month",
                                                             day_id="day",
                                                             hour_id="hour",
                                                             temp_id="temp")

hp_demand.to_csv("input/hp_demand.csv")

i = pd.date_range(start=hp_demand.index[0], end=hp_demand.index[-4], freq="H")
outdoor_temp = pd.DataFrame(index=i, data=list(df["temp"]), columns=["temp"]).interpolate(freq="15T")
outdoor_temp_15min = outdoor_temp.asfreq("15T")
outdoor_temp_15min = outdoor_temp_15min.interpolate()
outdoor_temp_15min.to_csv("input/outdoor_temp.csv")

