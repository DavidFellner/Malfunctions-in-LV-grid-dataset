import pandas as pd
import json
from datetime import datetime

import pvlib
from pvlib.location import Location

fn = "input/Dublin weather data 2018.json"

io = open(fn, "r")

wd = json.load(io)

df = pd.DataFrame(columns=["datetime", "wind_kph", "temp_c", "cloud"])

latitude = 53.11
longitude = -9.7

aran = Location(latitude, longitude, "UTC", 0, "Aran Islands")
times = pd.date_range(start="2018-1-1", end="2019-1-1", freq="1h", tz=aran.tz)
cs = aran.get_clearsky(times)

wind = pd.Series(index=times, name="wind_speed")
temp_air = pd.Series(index=times, name="temp_air")

sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

system = {'module': module, 'inverter': inverter,
          'surface_azimuth': 180, "surface_tilt": 35}


for day in wd["weather_data"]:

    hours = day["hour"]

    for hour in hours:

        date_time_obj = datetime.strptime(hour["time"], '%Y-%m-%d %H:%M')

        cloud = hour["cloud"]/100

        df = df.append({"datetime": date_time_obj,
                        "wind_kph": hour["wind_kph"],
                        "temp_c": hour["temp_c"],
                        "cloud": hour["cloud"]/100},
                  ignore_index=True)

        wind.loc[date_time_obj] = hour["wind_kph"]
        temp_air.loc[date_time_obj] = hour["temp_c"]


solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
dni_extra = pvlib.irradiance.get_extra_radiation(times)
airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
pressure = pvlib.atmosphere.alt2pres(0)
am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
tl = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
cs = pvlib.clearsky.ineichen(solpos['apparent_zenith'], am_abs, tl,
                             dni_extra=dni_extra, altitude=0)
aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'],
                           solpos['apparent_zenith'], solpos['azimuth'])

total_irrad = pvlib.irradiance.get_total_irradiance(
    system['surface_tilt'],
    system['surface_azimuth'],
    solpos['apparent_zenith'],
    solpos['azimuth'],
    cs['dni'], cs['ghi'], cs['dhi'],
    dni_extra=dni_extra,
    model='haydavies')

tcell = pvlib.temperature.sapm_cell(
    total_irrad['poa_global'],
    temp_air, wind,
    **temperature_model_parameters)

effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
    total_irrad['poa_direct'], total_irrad['poa_diffuse'],
    am_abs, aoi, module)

dc = pvlib.pvsystem.sapm(effective_irradiance, tcell, module)
ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)

ac.drop(ac.tail(1).index, inplace=True)
dc.drop(dc.tail(1).index, inplace=True) # in W

# reindex
new_index = pd.date_range(start=ac.index[0], end=ac.index[-1], freq="15T")
ac_15min = ac.reindex(new_index).interpolate()

# scale
p_mpp = module["Vmpo"] * module["Impo"]
target_max_power = -2.5


def scale_output(x, p_mpp=p_mpp, target_max_power=target_max_power):
    return (x / p_mpp) * target_max_power


ac_15min_scale = ac_15min.apply(scale_output).rename("ac_power_kW")

ac_15min_scale.to_csv("input/pv_profile.csv")

# print("ac sum:", sum(ac))
# print("ac max:", max(ac))
#
# print(dc)

# print("dc sum:", sum(dc))
# print("dc max:", max(dc))
