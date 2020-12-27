#!/usr/bin/env python
# coding: utf-8

"""
Combines faa/ntsb incident data by processing them into the same shape and concatenating 
them together
"""
import pandas as pd, pickle, numpy as np
from tqdm import tqdm as tqdm
from IPython import embed
from helper import fill_with_empty

ntsb = pd.read_csv('faa_ntsb_analysis/results/NTSB_AIDS_full_processed.csv')
ntsb['dataset'] = 'ntsb'
ntsb.rename({' Airport Code ': 'airport_code','ntsb_ Incident ': 'ntsb_incidents',\
        'ntsb_ Accident ': 'ntsb_accidents'}, axis = 1, inplace = True)
non_na_sel = ~ntsb['airport_code'].isna()
ntsb.loc[non_na_sel, 'airport_code'] = ntsb.loc[non_na_sel, 'airport_code'].str.upper()

faa_df = pd.read_csv('faa_ntsb_analysis/results/FAA_AIDS_full_processed.csv')
faa_df.rename({'tracon_code': 'airport_code'}, axis = 1, inplace = True)
faa_df['dataset'] = 'faa'

top_50_iata = set(pd.read_excel('datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
faa_df = faa_df.loc[faa_df['airport_code'].apply(lambda x: x in top_50_iata), :].copy()
ntsb = ntsb.loc[ntsb['airport_code'].apply(lambda x: x in top_50_iata), :].copy()

faa_df = fill_with_empty(faa_df, code_col='airport_code')
ntsb = fill_with_empty(ntsb, code_col='airport_code')

# find overlap
def create_tracon_dict(pd, col = 'tracon_code'):
    pd = pd.loc[~pd[col].isna(), :].copy()
    pd.set_index(['day', 'year', 'month', col], inplace = True)
    pd = pd.to_dict(orient = 'index')
    return pd, set(pd.keys())

ntsb_tracon_date = pd.read_csv('faa_ntsb_analysis/results/tracon_date_ntsb.csv')
ntsb_td_dict, ntsb_td_set = create_tracon_dict(ntsb_tracon_date, col = ' Airport Code ')

faa_tracon_date = pd.read_csv('faa_ntsb_analysis/results/tracon_date_faa.csv')
faa_td_dict, faa_td_set = create_tracon_dict(faa_tracon_date, col = 'tracon_code')

overlap = ntsb_td_set.intersection(faa_td_set)
overlap_records, overlap_nums = [], []
for key in overlap:
    num_ntsb, num_faa = ntsb_td_dict[key]['0'], faa_td_dict[key]['0']

    overlap_records.append(key)
    overlap_nums.append(min(num_ntsb, num_faa))

overlap = pd.DataFrame.from_records(overlap_records, \
        columns = ['day', 'year', 'month', 'tracon_code'])
overlap['num'] = overlap_nums

overlap = overlap.drop('day', axis = 1)\
        .groupby(['year', 'month', 'tracon_code'], as_index = False).sum()

ntsb.loc[ntsb['airport_code'].isna(), 'airport_code'] = 'nan'
faa_df.loc[faa_df['airport_code'].isna(), 'airport_code'] = 'nan'

index_cols = ['airport_code', 'year', 'month']
ntsb.set_index(index_cols, inplace = True)
faa_df.set_index(index_cols, inplace = True)

# combine and save
fin_df = pd.concat([ntsb, faa_df], axis = 0, sort = False).groupby(level = list(range(3))).sum()
fin_df.reset_index(inplace = True)
fin_df.loc[fin_df['airport_code'] == 'nan', 'airport_code'] = np.nan

fin_df['faa_ntsb_overlap'] = 0
for idx, row in overlap.iterrows():
    sel = (fin_df['airport_code'] == row['tracon_code']) & \
            (fin_df['year'] == row['year']) & \
            (fin_df['month'] == row['month'])
    fin_df.loc[sel, 'faa_ntsb_overlap'] = row['num']

assert(fin_df.drop_duplicates(index_cols).shape[0] == fin_df.shape[0])

fin_df['NTSB_FAA_incidents_total'] = fin_df['faa_incidents'] + fin_df['ntsb_incidents'] + \
        fin_df['ntsb_accidents']
fin_df['NTSB_FAA_incidents_total_nodups'] = fin_df['NTSB_FAA_incidents_total'] - fin_df['faa_ntsb_overlap']

# save results
fin_df.to_csv('results/airport_month_events.csv')
pickle.dump(fin_df['airport_code'].unique(), open('results/unique_airport_code_ntsb_faa.pckl', 'wb'))
