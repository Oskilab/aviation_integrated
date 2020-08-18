#!/usr/bin/env python
# coding: utf-8

"""
Combines faa/ntsb incident data by processing them into the same shape and concatenating 
them together
"""
import pandas as pd
from tqdm import tqdm as tqdm
from IPython import embed

ntsb = pd.read_csv('aviation_data_huiyi/results/NTSB_AIDS_full_output_new.csv')
faa = pd.read_csv('faa/results/faa_incidents.csv', index_col = 0)

ntsb.rename({'airportcode_new': 'airport_code', 'airportname_new': 'airport_name', \
        ' Investigation Type _ Accident ': 'ntsb_accidents', \
        ' Investigation Type _ Incident ': 'ntsb_incidents'},
           axis = 1, inplace = True)
ntsb['dataset'] = 'ntsb'

# preprocess faa into correct shape. faa_incidents.csv had each unique month/year as a column
# but we change the dataframe so that each row only has one particular month/year
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
cols = list(faa.columns)[2:]

df_dict = {'airport_code': [], 'airport_name': [], 'year': [], 'month': [], 'faa_incidents': []}
for idx in tqdm(range(1, faa.shape[0])):
    row = faa.iloc[idx, :]
    for col in cols:
        if row[col] > 0:
            df_dict['airport_code'].append(row['iata_code'])
            df_dict['airport_name'].append(row['clean_name'])
            
            yr = int(col.split("-")[1])
            if yr >= 78:
                yr += 1900
            else:
                yr += 2000
                
            month = months.index(col.split("-")[0]) + 1
            df_dict['year'].append(yr)
            df_dict['month'].append(month)
            df_dict['faa_incidents'].append(row[col])

faa_df = pd.DataFrame(df_dict)
faa_df['dataset'] = 'faa'

# combine and save
fin_df = pd.concat([ntsb, faa_df], axis = 0, sort = False)
fin_df = fin_df[['airport_code', 'airport_name', 'year', 'month', 'ntsb_accidents',\
        'ntsb_incidents', 'faa_incidents', 'dataset']]
fin_df.fillna(0, inplace = True)
fin_df.to_csv('results/airport_month_events.csv')
