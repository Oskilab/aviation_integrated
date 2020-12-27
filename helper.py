from tqdm import tqdm
import pandas as pd
import itertools

def tracon_month_split(x):
    """
    This extracts the tracon_code, month and year from a given string (x).
    @param: x (string) with format 'tracon_code year/month'
    @returns: tuple of (tracon_code, year, month), w/types (str, float, float)
    """
    return get_tracon(x), get_month(x), get_year(x)

def get_tracon(x):
    """
    Extracts tracon_code from a given string (x)
    @param: x (string) with format 'tracon_code year/month'
    @returns: tracon_code (str)
    """
    split_x = str(x).split()
    if len(split_x) > 2:
        return " ".join(split_x[:-1])
    else:
        return split_x[0]

def get_year(x):
    """
    Extracts year from a given string (x)
    @param: x (string) with format 'tracon_code year/month'
    @returns: year (float), nan if format incorrect
    """
    try:
        return int(float(str(x).split()[-1].split("/")[0]))
    except ValueError:
        return np.nan

def get_month(x):
    """
    Extracts month from a given string (x)
    @param: x (string) with format 'tracon_code year/month'
    @returns: month (float), nan if format incorrect
    """
    try:
        return int(float(str(x).split()[-1].split("/")[1]))
    except ValueError:
        return np.nan

def fill_with_empty(df, code_col='airport_code', date_range=range(1988,2020)):
    """
    Fills df with empty_rows. It calculates all unique airport codes within
    the df, and then for every combination of airport code and date, we add
    an empty row if the original dataframe did not have a row corresponding
    to that airport code and date. This is called in combine.py
    @param: df (pd.DataFrame) dataset we are analyzing
    @param: code_col (str) column that indicates the airport_code
    @param: date_range (range) this is range of years, currently, we look 
        at dates between 1988 and 2020
    @returns: adjusted df with empty rows
    """
    assert('year' in df.columns and 'month' in df.columns)
    all_combs = set()
    for idx, row in df[[code_col, 'year', 'month']].drop_duplicates().iterrows():
        all_combs.add((row[code_col], int(row['year']), int(row['month'])))

    unique_codes = df[code_col].unique()
    empty_row = pd.Series(index = df.columns)
    new_rows = []
    for code, year, month in tqdm(itertools.product(unique_codes, date_range, range(1,13)), \
            total = unique_codes.shape[0] * len(date_range) * 12):
        comb = (code, year, month)
        if comb not in all_combs:
            copy = empty_row.copy()
            copy[code_col] = code
            copy['year'] = year
            copy['month'] = month
            new_rows.append(copy)
    new_rows = pd.DataFrame.from_records(new_rows)
    return pd.concat([df, new_rows], axis = 0)
