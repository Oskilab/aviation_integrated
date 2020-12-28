"""
Helper functions
"""
import itertools
from tqdm import tqdm
import pandas as pd
import numpy as np

def tracon_month_split(trcn_mnth):
    """
    This extracts the tracon_code, month and year from a given string (x).
    @param: trcn_mnth (string) with format 'tracon_code year/month'
    @returns: tuple of (tracon_code, year, month), w/types (str, float, float)
    """
    return get_tracon(trcn_mnth), get_month(trcn_mnth), get_year(trcn_mnth)

def get_tracon(trcn_mnth):
    """
    Extracts tracon_code from a given string (x)
    @param: trcn_mnth (string) with format 'tracon_code year/month'
    @returns: tracon_code (str)
    """
    split_x = str(trcn_mnth).split()
    if len(split_x) > 2:
        return " ".join(split_x[:-1])
    return split_x[0]

def get_year(trcn_mnth):
    """
    Extracts year from a given string (trcn_mnth)
    @param: trcn_mnth (string) with format 'tracon_code year/month'
    @returns: year (float), nan if format incorrect
    """
    try:
        return int(float(str(trcn_mnth).split()[-1].split("/")[0]))
    except ValueError:
        return np.nan

def get_month(trcn_mnth):
    """
    Extracts month from a given string (trcn_mnth)
    @param: trcn_mnth (string) with format 'tracon_code year/month'
    @returns: month (float), nan if format incorrect
    """
    try:
        return int(float(str(trcn_mnth).split()[-1].split("/")[1]))
    except ValueError:
        return np.nan

def fill_with_empty(dataset, code_col='airport_code', date_range=range(1988, 2020)):
    """
    Fills dataset with empty_rows. It calculates all unique airport codes within
    the dataset, and then for every combination of airport code and date, we add
    an empty row if the original dataframe did not have a row corresponding
    to that airport code and date. This is called in combine.py
    @param: dataset (pd.DataFrame) dataset we are analyzing
    @param: code_col (str) column that indicates the airport_code
    @param: date_range (range) this is range of years, currently, we look
        at dates between 1988 and 2020
    @returns: adjusted dataset with empty rows
    """
    assert('year' in dataset.columns and 'month' in dataset.columns)
    all_combs = set()
    for _, row in dataset[[code_col, 'year', 'month']].drop_duplicates().iterrows():
        all_combs.add((row[code_col], int(row['year']), int(row['month'])))

    unique_codes = dataset[code_col].unique()
    empty_row = pd.Series(index=dataset.columns)
    new_rows = []
    for code, year, month in tqdm(itertools.product(unique_codes, date_range, range(1, 13)), \
            total=unique_codes.shape[0] * len(date_range) * 12):
        comb = (code, year, month)
        if comb not in all_combs:
            copy = empty_row.copy()
            copy[code_col] = code
            copy['year'] = year
            copy['month'] = month
            new_rows.append(copy)
    new_rows = pd.DataFrame.from_records(new_rows)
    return pd.concat([dataset, new_rows], axis=0)
