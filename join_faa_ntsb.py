#!/usr/bin/env python
# coding: utf-8
"""
Combines faa/ntsb incident data by processing them into the same shape and concatenating
them together. Also calculates overlap between faa/ntsb incident/accident dataset.
"""
import pickle

import pandas as pd
import numpy as np

index_cols = ['airport_code', 'year', 'month']

def load_ntsb_inc_acc_ds():
    """
    This loads the ntsb_aids dataset from faa_ntsb_analysis. Function is read as
    load ntsb incident/accident dataset.
    @returns: ntsb (pd.DataFrame) from faa_ntsb_analysis subdirectory.
    """
    ntsb = pd.read_csv('faa_ntsb_analysis/results/NTSB_AIDS_full_processed.csv')
    ntsb['dataset'] = 'ntsb' # track which dataset
    ntsb.rename({' Airport Code ': 'airport_code', 'ntsb_ Incident ': 'ntsb_incidents',\
            'ntsb_ Accident ': 'ntsb_accidents'}, axis=1, inplace=True)
    non_na_sel = ~ntsb['airport_code'].isna()
    ntsb.loc[non_na_sel, 'airport_code'] = ntsb.loc[non_na_sel, 'airport_code'].str.upper()
    return ntsb

def load_faa_inc_acc_ds():
    """
    This loads the faa_aids dataset from faa_ntsb_analysis. Function is read as
    load faa incident/accident dataset.
    @returns: faa (pd.DataFrame) from faa_ntsb_analysis subdirectory.
    """
    faa_df = pd.read_csv('faa_ntsb_analysis/results/FAA_AIDS_full_processed.csv')
    faa_df.rename({'tracon_code': 'airport_code'}, axis=1, inplace=True)
    faa_df['dataset'] = 'faa'
    return faa_df

def filter_top_50(inc_acc_ds, top_50_iata):
    """
    Only selects rows with airport codes from top50 iata codes
    @param: inc_acc_ds (pd.DataFrame), either ntsb_inc_acc_ds or faa_inc_acc_ds
    @param: top_50_iata (set of str), contains set of airport codes of top 50
    @returns: subset of inc_acc_ds
    """
    return inc_acc_ds.loc[inc_acc_ds['airport_code'].apply(lambda x: x in top_50_iata), :].copy()

def create_tracon_dict(inc_df, col='tracon_code'):
    """
    Creates a dictionary that maps from (day, year, month, tracon_code) -> number of rows
    in the given df, and a set that contains all combinations of (day, year, month, tracon_code).
    The results are eventually utilized to develop a df of overlapping incident/accidents
    @param: inc_df (pd.DataFrame), either ntsb_inc_date_ds or faa_inc_date_ds (see
        load_faa_date_ds() or load_ntsb_date_ds())
    @param: col (str) column we are analyzing either tracon_code or Airport Code,
        depending on which inc_df we are analyzing
    @returns: dict[(day, year, month, tracon_code)] -> dict['0'] -> # rows
        in other words, this is a nested dictionary, where each value_dictionary contains the
        key '0' and that key maps to the number of rows in inc_acc_ds
    @returns: set(all combs of (day, year, month, tracon_code) in inc_df)
    """
    inc_df = inc_df.loc[~inc_df[col].isna(), :].copy()
    inc_df.set_index(['day', 'year', 'month', col], inplace=True)
    inc_df = inc_df.to_dict(orient='index')
    return inc_df, set(inc_df.keys())

def load_faa_date_ds():
    """
    This loads the dataframe consisting of (day, year, month, tracon_code) and the number of
    times that combination occurs within the faa_inc_date_ds. This is generated in the file
    faa_ntsb_analysis/find_faa_code.py. We then call create_tracon_dict to generate info
    about dates and the number of incidents/accidents that occur on each date.
    @returns: dict[(day, year, month, tracon_code)] -> dict['0'] -> # rows
        in other words, this is a nested dictionary, where each value_dictionary contains the
        key '0' and that key maps to the number of rows in the inc_acc_ds
    @returns: set(all combs of (day, year, month, tracon_code) in faa_inc_dat_ds)
    """
    faa_tracon_date = pd.read_csv('faa_ntsb_analysis/results/tracon_date_faa.csv')
    faa_td_dict, faa_td_set = create_tracon_dict(faa_tracon_date, col='tracon_code')
    return faa_td_dict, faa_td_set

def load_ntsb_date_ds():
    """
    This loads the dataframe consisting of (day, year, month, tracon_code) and the number of
    times that combination occurs within the ntsb_inc_date_ds. This is generated in the file
    faa_ntsb_analysis/find_ntsb_code.py. We then call create_tracon_dict to generate info
    about dates and the number of incidents/accidents that occur on each date.
    @returns: dict[(day, year, month, tracon_code)] -> dict['0'] -> # rows
        in other words, this is a nested dictionary, where each value_dictionary contains the
        key '0' and that key maps to the number of rows in the inc_acc_ds
    @returns: set(all combs of (day, year, month, tracon_code) in ntsb_inc_dat_ds)
    """
    ntsb_tracon_date = pd.read_csv('faa_ntsb_analysis/results/tracon_date_ntsb.csv')
    ntsb_td_dict, ntsb_td_set = create_tracon_dict(ntsb_tracon_date, col=' Airport Code ')
    return ntsb_td_dict, ntsb_td_set

def calculate_overlap(ntsb_td_dict, ntsb_td_set, faa_td_dict, faa_td_set):
    """
    This calculates the dataframe overlap, where each row contains year/month/tracon_code
    and the number of times there were duplicate events between faa/ntsb inc/acc ds.
    @param: ntsb_td_dict (dict[(day, year, month, tracon_code)] -> dict['0'] -> # rows)
        see description above
    @param: ntsb_td_set(all combs of (day, year, month, tracon_code) in ntsb_inc_dat_ds)
    @param: faa_td_dict (dict[(day, year, month, tracon_code)] -> dict['0'] -> # rows)
        see description above
    @param: faa_td_set(all combs of (day, year, month, tracon_code) in faa_inc_dat_ds)
    @returns: overlap (pd.DataFrame), with cols = [year, month, tracon_code, num]
    """
    # generate records of overlap
    overlap = ntsb_td_set.intersection(faa_td_set)
    overlap_records, overlap_nums = [], []
    for key in overlap:
        num_ntsb, num_faa = ntsb_td_dict[key]['0'], faa_td_dict[key]['0']

        overlap_records.append(key)
        overlap_nums.append(min(num_ntsb, num_faa))

    # create df and group by tracon_month
    overlap = pd.DataFrame.from_records(overlap_records, \
            columns=['day', 'year', 'month', 'tracon_code'])
    overlap['num'] = overlap_nums

    overlap = overlap.drop('day', axis=1)\
            .groupby(['year', 'month', 'tracon_code'], as_index=False).sum()
    return overlap

def postprocess_ds(inc_acc_ds):
    """
    This post-processes an inc/acc ds by rreplacing nan airport codes with nan string
    and resets index to [airport_code, year, month]
    @param: inc_acc_ds (pd.DataFrame) inc_acc_ds
    @returns: inc_acc_ds (pd.DataFrame) inc_acc_ds post-processed
    """
    inc_acc_ds.loc[inc_acc_ds['airport_code'].isna(), 'airport_code'] = 'nan'
    inc_acc_ds.set_index(index_cols, inplace=True)
    return inc_acc_ds

def combine_faa_ntsb(ntsb, faa_df):
    """
    This combines ntsb_inc_acc_ds with faa_inc_acc_ds and then groups by tracon_month
    and sums over tracon_months to calculate the number of incidents for that tracon_month
    @param: ntsb (pd.DataFrame), ntsb_inc_acc_ds
    @param: faa_df (pd.DataFrame), faa_inc_acc_ds
    @returns: fin_df (pd.DataFrame), combined inc_acc_ds
    """
    fin_df = pd.concat([ntsb, faa_df], axis=0, sort=False).groupby(level=list(range(3))).sum()
    fin_df.reset_index(inplace=True)
    fin_df.loc[fin_df['airport_code'] == 'nan', 'airport_code'] = np.nan
    return fin_df

def create_overlap_col(fin_df, overlap):
    """
    This creates a new column called faa_ntsb_overlap which calculates the number of
    overlapping incidents/accidents between ntsb/faa.
    @param: fin_df (pd.DataFrame), combined incident_accident dataset
    @param: overlap (pd.DataFrame), generated from calculate_overlap (see description)
    """
    fin_df['faa_ntsb_overlap'] = 0
    for _, row in overlap.iterrows():
        sel = (fin_df['airport_code'] == row['tracon_code']) & \
                (fin_df['year'] == row['year']) & \
                (fin_df['month'] == row['month'])
        fin_df.loc[sel, 'faa_ntsb_overlap'] = row['num']
    assert fin_df.drop_duplicates(index_cols).shape[0] == fin_df.shape[0]
    return fin_df

def main():
    """
    Joins FAA/NTSB incident/accident datasets
    """
    ntsb = load_ntsb_inc_acc_ds()
    faa_df = load_faa_inc_acc_ds()

    # filter top50
    top_50_iata = \
            set(pd.read_excel('datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    faa_df, ntsb = filter_top_50(faa_df, top_50_iata), filter_top_50(ntsb, top_50_iata)

    faa_td_dict, faa_td_set = load_faa_date_ds()
    ntsb_td_dict, ntsb_td_set = load_ntsb_date_ds()

    overlap = calculate_overlap(ntsb_td_dict, ntsb_td_set, faa_td_dict, faa_td_set)

    ntsb = postprocess_ds(ntsb)
    faa_df = postprocess_ds(faa_df)

    # combine datasets and compute overlap columns
    fin_df = combine_faa_ntsb(ntsb, faa_df)
    fin_df = create_overlap_col(fin_df, overlap)
    fin_df['NTSB_FAA_incidents_total'] = fin_df['faa_incidents'] + fin_df['ntsb_incidents'] + \
            fin_df['ntsb_accidents']
    fin_df['NTSB_FAA_incidents_total_nodups'] = fin_df['NTSB_FAA_incidents_total'] - \
            fin_df['faa_ntsb_overlap']

    # save results
    fin_df.to_csv('results/airport_month_events.csv')
    pickle.dump(fin_df['airport_code'].unique(), \
            open('results/unique_airport_code_ntsb_faa.pckl', 'wb'))

if __name__ == "__main__":
    main()
