from itertools import product
from tqdm import tqdm
import pandas as pd
def tracon_month_split(x):
    return get_tracon(x), get_month(x), get_year(x)

def get_tracon(x):
    split_x = str(x).split()
    if len(split_x) > 2:
        return " ".join(split_x[:-1])
    else:
        return split_x[0]

def get_year(x):
    try:
        return int(float(str(x).split()[-1].split("/")[0]))
    except ValueError:
        return np.nan

def get_month(x):
    try:
        return int(float(str(x).split()[-1].split("/")[1]))
    except ValueError:
        return np.nan

def fill_with_empty(df, code_col = 'airport_code', date_range = range(1988,2020)):
    assert('year' in df.columns and 'month' in df.columns)
    all_combs = set()
    for idx, row in df[[code_col, 'year', 'month']].drop_duplicates().iterrows():
        all_combs.add((row[code_col], int(row['year']), int(row['month'])))
    # all_combs = set(df[[code_col, 'year', 'month']].apply(lambda x: (x[0], x[1], x[2])))

    unique_codes = df[code_col].unique()
    empty_row = pd.Series(index = df.columns)
    new_rows = []
    for code, year, month in product(unique_codes, date_range, range(1,13)):
        comb = (code, year, month)
        if comb not in all_combs:
            copy = empty_row.copy()
            copy[code_col] = code
            copy['year'] = year
            copy['month'] = month
            new_rows.append(copy)
    new_rows = pd.DataFrame.from_records(new_rows)
    return pd.concat([df, new_rows], axis = 0)
