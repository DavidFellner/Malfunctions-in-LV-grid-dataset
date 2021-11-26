import pandas as pd
import os

file_path = os.path.join(os.path.dirname(__file__), os.pardir)

ev_characteristics = {'name': ['egolf', 'zoe', 'i3', 'leaf', 'ioniq', 'models'],
                      'consumption': [16.8, 15.4, 16.1, 16.5, 14.4, 18.4],
                      'phases': [2, 3, 1, 1, 1, 3],
                      'max_current': [16, 32, 32, 32, 32, 32],
                      'battery_size': [35.8, 41, 37.9, 62, 28, 100]}

char_df = pd.DataFrame(data=ev_characteristics)
char_df.to_csv(os.path.join(file_path, 'ev_characteristics.csv'))

# verbrauchswerte nach oem angabe da diese schwieriger sind für das netz!
# hier anders als im deliverable!
