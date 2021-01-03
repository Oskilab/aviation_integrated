"""
Loads FAA incident/accident dataset and creates tracon_code column
"""
import re
import pickle

from tqdm import tqdm
from requests import HTTPError

import pandas as pd
import numpy as np

def load_faa_data():
    """
    This loads in the FAA incident dataset and does some preprocessing to column names
    @returns: pd.DataFrame of FAA incident dataset.
    """
    full = pd.read_csv('datasets/FAA_AIDS_full.csv')
    new = pd.read_csv('datasets/FAA_AIDS_addition.csv')

    # rename columns
    rename_dict = {}
    for col in new.columns:
        rename_dict[col] = col.lower().replace(" ", "")

    new.rename(rename_dict, axis=1, inplace=True)
    return pd.concat([full, new], axis=0, ignore_index=True, sort=False)

def get_month(date):
    """
    This extracts the month from the column 'localeventdate' in faa incident dataset.
    @param: date (str)
    @returns: month (int/float), nan if it cannot be extracted
    """
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',\
            'OCT', 'NOV', 'DEC']
    split_date = date.split("-")
    if len(split_date) != 3:
        return np.nan
    month_str = split_date[1]
    if month_str not in months:
        return np.nan
    return months.index(month_str) + 1

def get_year(date):
    """
    This extracts the year from the column 'localeventdate' in faa incident dataset.
    @param: date (str)
    @returns: year (int/float), nan if it cannot be extracted
    """
    split_date = date.split("-")
    if len(split_date) != 3:
        return np.nan
    year_str = split_date[2]
    try:
        year = int(year_str)
        if (year >= 78) and (year < 100):
            return year + 1900
        return year + 2000
    except ValueError:
        return np.nan

def get_day(date):
    """
    This extracts the day of the month from the column 'localeventdate' in faa incident
    dataset.
    @param: date (str)
    @returns: day of the month (int/float), nan if it cannot be extracted
    """
    split_date = date.split("-")
    if len(split_date) != 3:
        return np.nan
    day_str = split_date[0]
    try:
        day = int(day_str)
        return day
    except ValueError:
        return np.nan

def process_date(faa_inc):
    """
    This processes the dates of the faa incident dataset.
    @param: faa_inc (pd.DataFrame) faa incident dataset
    @returns: faa_inc (pd.DataFrame) faa incident dataset w/added time columns
    """
    faa_inc['month'] = faa_inc['localeventdate'].apply(get_month)
    faa_inc['year'] = faa_inc['localeventdate'].apply(get_year)
    faa_inc['day'] = faa_inc['localeventdate'].apply(get_day)
    faa_inc = faa_inc.loc[(faa_inc['year'] >= 1988) & (faa_inc['year'] < 2020)].copy()
    return faa_inc

def process_location(faa_inc):
    """
    This adds new columns to the dataframe regarding the location (state/city)
    @param: faa_inc (pd.DataFrame) faa incident dataset
    @returns: faa_inc (pd.DataFrame) faa incident dataset w/added location columns
    """
    # load us state abbreviations
    us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col=0)
    us_state_abbrev = us_state_abbrev.to_dict()['full']

    faa_inc['event_fullstate'] = faa_inc['eventstate'] \
            .apply(lambda x: us_state_abbrev.get(x, ''))
    faa_inc['eventcity'] = faa_inc['eventcity'].str.lower()
    return faa_inc

# needed for process_faa_name
replace = {'rgnl': ['regional'], 'fld':['field'], 'intl': ['international'], \
            'muni': ['municipal'], 'univ.' :['university'],\
            'iap': ['international', 'airport'], 'afb': ['air', 'force', 'base']}

def replace_with(string_input, replace_dict={}):
    """
    Helper function that replaces a word with its full form. The output however
    should be a list of words.
    @param: string_input (str) word
    @param: replace (dict: str->list of str)
    @returns: list of replaced version of the word
    """
    return replace_dict.get(string_input, [string_input])

def add_to_arr_replace(string_input, arr):
    """
    Add the replaced version of s to arr
    @param: string_input (str) word
    @param: arr (list) of final words
    """
    for replace_elem in replace_with(string_input, replace):
        arr.append(replace_elem)

def process_faa_name(airport_name):
    """
    This process the name of an airport (cleaning extraneous characters, etc.)
    @param: airport_name (str) name of an airport
    @returns: str of cleaned version of airport_name
    """
    def process_word(fin, word):
        """
        This processes a word inside the name of an airport. It uses regex to
        separate words that are separated by - or / (eg hello/world) and also
        replaces some common abbreviations by their full form utilizing the
        replace dictionary above.
        @param: fin (list) of final result
        @param: word (str)
        """
        if word == "-":
            return
        # pattern for processing words
        reg_dash = '([a-z]{1,})[-/]{1}([a-z]{1,})'
        pat = re.compile(reg_dash)

        # stripping unimportant chars
        if word.startswith("(") or word.startswith("/"):
            word = word[1:]
        if word.endswith(")") or word.endswith("/") or word.endswith(",") or word.endswith("-"):
            word = word[:-1]

        # add to fin
        reg_res = pat.match(word)
        if reg_res is not None:
            add_to_arr_replace(reg_res.group(1), fin)
            add_to_arr_replace(reg_res.group(2), fin)
        else:
            add_to_arr_replace(word, fin)

    airport_name = str(airport_name).lower()
    fin = []
    for word in airport_name.split(" "):
        elem_arr = replace_with(word, replace)
        for elem in elem_arr:
            process_word(fin, elem)
    return " ".join(fin)

def fill_in_handcoded(faa_df):
    """
    This fills in the faa incident dataset with IATA codes by utilizing the name of airport
    city and state, and a handcoded dataset with airport names/city/states and their corresponding
    airport codes
    @param: faa_df (pd.DataFrame) faa incident dataset
    @returns: faa_df (pd.DataFrame) with filled in iata codes
    """
    handcoded = pd.read_csv('datasets/not_matched_full_v1.csv', index_col=0)
    handcoded.drop(['Unnamed: 8'], axis=1, inplace=True)
    handcoded.rename({'Unnamed: 7': 'tracon_code'}, axis=1, inplace=True)
    handcoded = handcoded.loc[~handcoded['tracon_code'].isna(), :].copy()

    for _, row in tqdm(handcoded.iterrows(), total=handcoded.shape[0], desc="handcoded"):
        if row['tracon_code'] != '?':
            sel = (faa_df['eventairport_conv'] == row['eventairport_conv']) & \
                     (faa_df['eventcity'] == row['eventcity']) & \
                     (faa_df['eventstate'] == row['eventstate'])

            faa_df.loc[sel, 'tracon_code'] = row['tracon_code']

    return faa_df

def fill_in_iata(faa_inc, full_matched_pd, wiki_search_found_df):
    """
    Given the dataframe of matched wikipedia tracon_codes and a dataframe of matched tracon_codes
    from querying wikipedia (the latter is made via requests to wikipedia, the former is scraped),
    fill in the faa incident dataset with the corresponding tracon_code.
    @param: faa_inc (pd.DataFrame) faa incident dataset
    @param: full_matched_pd (pd.DataFrame) dataframe of matched airport names (from df),
        and their corresponding airport codes found on a wikipedia table of airport codes
    @param: wiki_search_found_df (pd.DataFrame) dataframe of matched airport names (from df),
        and their corresponding airport codes found by querying the wikipedia website
    @returns: df (pd.DataFrame) faa incident dataset with the tracon codes filled in
    """
    name_to_iata = {}
    for idx, row in full_matched_pd.iterrows():
        name_to_iata[row['eventairport_conv']] = row['wiki_IATA']
    for idx, row in wiki_search_found_df.iterrows():
        name_to_iata[idx] = row['iata']

    faa_inc['tracon_code'] = np.nan
    faa_inc['tracon_code'] = faa_inc['eventairport_conv'].apply(lambda x: name_to_iata[x] \
            if x in name_to_iata else np.nan)
    faa_inc = fill_in_handcoded(faa_inc)
    return faa_inc

def fill_in_handcode_iata(faa_inc):
    """
    This utilizes a handcoded dictionary that maps between airport name to matching IATA code.
    @param: faa_inc (pd.DataFrame) dataframe of FAA incident/accident dataset
    @returns: processed faa_inc with handcoded IATA codes
    """
    handcode_iata_dict_df = pd.read_excel('datasets/FAA_key.xlsx')
    handcode_iata_dict = {}
    for idx, row in handcode_iata_dict_df.set_index('FAA_key').iterrows():
        handcode_iata_dict[idx] = row['tracon_key']
    faa_inc['tracon_code'] = faa_inc['eventairport'] \
            .apply(lambda x: handcode_iata_dict.get(x, np.nan))
    return faa_inc

def post_process_results(faa_inc):
    """
    This post processes the results by grouping by tracon_month and then counting the number
    of rows (and setting a new column faa_incidents equal to the number of rows). Also deals
    with nans/none.
    @param: faa_inc (pd.DataFrame) dataframe at the end of the pipeline
    @returns: processed faa_inc (pd.DataFrame)
    """
    # hack around groupby ignoring nan values
    faa_inc['tracon_code'] = faa_inc['tracon_code'].fillna('nan')

    tracon_date = pd.DataFrame(faa_inc.groupby(['day', 'month', 'year', 'tracon_code']).size())
    tracon_date.to_csv('results/tracon_date_faa.csv')

    cols = ['tracon_code', 'month', 'year', 'eventtype']
    faa_inc = faa_inc[cols].groupby(cols[:-1]).count().\
            rename({cols[-1]: 'faa_incidents'}, axis=1).reset_index()
    faa_inc['tracon_code'] = faa_inc['tracon_code'].str.replace("none", "")

    # deal with na values
    faa_inc.loc[faa_inc['tracon_code'] == 'nan', 'tracon_code'] = np.nan
    faa_inc.loc[faa_inc['tracon_code'] == 'none', 'tracon_code'] = np.nan
    return faa_inc

def main(verbose=True):
    # load datasets
    full = load_faa_data()
    if verbose:
        print('total original number of rows', full.shape[0])

    # process date
    full = process_date(full)
    if verbose:
        print('number of rows, filtered by date', full.shape[0])

    full = process_location(full)

    # process FAA airport name in order to utilize it for wikipedia traocn
    # searches
    full['eventairport_conv'] = full['eventairport'].apply(process_faa_name)

    # fill in dataset with matched iata codes (found via airport name above)
    full = fill_in_handcode_iata(full)
    full = fill_in_handcoded(full)

    if verbose:
        print('non empty tracon code rows', full.loc[~full['tracon_code'].isna()].shape[0])

    full = post_process_results(full)
    full.to_csv('results/FAA_AIDS_full_processed.csv', index=False)

if __name__ == "__main__":
    main()
