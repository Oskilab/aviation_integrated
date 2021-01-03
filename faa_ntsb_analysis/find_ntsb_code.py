"""
Loads NTSB incident/accident dataset and creates tracon_code column
"""
import re
import pickle
from urllib.error import HTTPError

from tqdm import tqdm
import numpy as np
import pandas as pd

from common_funcs import get_city, get_state, get_country

# load us state abbreviations
us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col=0)
us_state_abbrev = us_state_abbrev.to_dict()['full']

us_to_abbrev = {v: k for k, v in us_state_abbrev.items()}

def process_code(ntsb_inc):
    """
    Processes the Airport Code field in the NTSB incident/accident dataset by removing na values
    and turning everything to uppercase
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @returns: ntsb_inc (pd.DataFrame) ntsb incident dataset w/filtered Airport Code
    """
    invalid_codes = ['n/a', 'none', 'na']
    for inv_code in invalid_codes:
        ntsb_inc[' Airport Code '] = ntsb_inc[' Airport Code '].str.replace(inv_code, "")
    nonna_code_sel = ~(ntsb_inc[' Airport Code '].isna())
    ntsb_inc.loc[nonna_code_sel, ' Airport Code '] = \
            ntsb_inc.loc[nonna_code_sel, ' Airport Code '].str.upper()
    ntsb_inc.loc[ntsb_inc[' Airport Code '].isna(), ' Airport Code '] = ''
    return ntsb_inc

def process_times(ntsb_inc, verbose=True):
    """
    Processes the date in the NTSB incident/accident dataset.
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @returns: ntsb_inc (pd.DataFrame) ntsb incident dataset w/processed date
    """
    ntsb_inc['year'] = ntsb_inc[' Event Date '].str.split("/").apply(lambda x: x[2]).apply(int)
    ntsb_inc['month'] = ntsb_inc[' Event Date '].str.split("/").apply(lambda x: x[0]).apply(int)
    ntsb_inc['day'] = ntsb_inc[' Event Date '].str.split("/").apply(lambda x: x[1]).apply(int)
    if verbose:
        print('full dataset size', ntsb_inc.shape[0])
    ntsb_inc = ntsb_inc.loc[(ntsb_inc['year'] >= 1988) & (ntsb_inc['year'] < 2020), :].copy()
    return ntsb_inc

def print_preliminary_stats(ntsb_inc, verbose=True):
    """
    Prints preliminary statistics on the NTSB incident/accident dataset.
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    """
    if verbose:
        print('airport code missing', ntsb_inc.loc[ntsb_inc[' Airport Code '] != ''].shape[0])
        print('airport name missing', ntsb_inc.loc[ntsb_inc[' Airport Name '] != ''].shape[0])
        print('airport code or name missing', ntsb_inc.loc[(ntsb_inc[' Airport Name '] != '') | \
                (ntsb_inc[' Airport Code '] != '')].shape[0])
        print('airport name and no code missing', \
                ntsb_inc.loc[(ntsb_inc[' Airport Name '] != '') & \
                (ntsb_inc[' Airport Code '] == '')].shape[0])

def process_ntsb_name(airport_name):
    """
    This processes the airport name field in the NTSB incident/accident dataset
    @param: airport_name (str) name of the airport
    @returns: processed name of the airport
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
        if word not in ["airport", "-"]:
            return None

        reg_dash = '([a-z]{1,})[-/]{1}([a-z]{1,})'
        pat = re.compile(reg_dash)

        if word.startswith("(") or word.startswith("/"):
            word = word[1:]
        if word.endswith(")") or word.endswith("/") or word.endswith(",") or word.endswith("-"):
            word = word[:-1]

        # add to fin
        reg_res = pat.match(word)
        if reg_res is not None:
            fin.append(reg_res.group(1))
            fin.append(reg_res.group(2))
        else:
            fin.append(word)
        return None

    airport_name = str(airport_name).lower()
    fin = []
    for elem in airport_name.split(" "):
        process_word(fin, elem)
    return " ".join(fin)

def process_location(ntsb_inc):
    """
    This processes the location information in the NTSB incident/accident dataset
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @returns: ntsb_inc (pd.DataFrame) ntsb incident dataset w/filtered location
    """
    ntsb_inc['eventcity'] = ntsb_inc[' Location '].apply(get_city)
    ntsb_inc['event_fullstate'] = ntsb_inc[' Location '].apply(get_state)
    ntsb_inc['event_country'] = ntsb_inc[' Location '].apply(get_country)
    return ntsb_inc

def save_discarded(ntsb_inc):
    """
    This saves the part of the NTSB incident/accident dataset that have no airport codes
    nor airport names to a separate csv file. Then, we utilize the rest of the dataset
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset w/airport codes or names
    """
    code_and_name_empty = (ntsb_inc[' Airport Code '] == '') & (ntsb_inc[' Airport Name '] == '')
    ntsb_inc.loc[code_and_name_empty, :].to_csv('results/discarded_ntsb.csv')
    return ntsb_inc.loc[~code_and_name_empty, :].copy()

def conv_to_float(str_input):
    """
    Helper function that converts a string to a float (nan if it's an empty string)
    @param: str_input (str)
    @param: float version of str_input
    """
    if str_input == '':
        return np.nan
    return round(float(str_input.strip()), 5)

def process_lat_lng(ntsb_inc):
    """
    Processes the latitude/longitude fields in the NTSB incident/accident dataset.
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @returns: ntsb_inc (pd.DataFrame) ntsb incident dataset w/processed latitude/longitude
    """
    ntsb_inc[' Latitude '] = ntsb_inc[' Latitude '].apply(conv_to_float)
    ntsb_inc[' Longitude '] = ntsb_inc[' Longitude '].apply(conv_to_float)
    return ntsb_inc

def preprocess_data(ntsb_inc, verbose=True):
    """
    Processes the NTSB incident/accident dataset by calling the above functions. It processes
    airport name, airport code, location, and latitude/longitude info.
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @param: ntsb_inc (pd.DataFrame) processed ntsb incident dataset
    """
    strip_cols = [' Airport Code ', ' Airport Name ', ' Latitude ', ' Longitude ']
    for col in strip_cols:
        ntsb_inc[col] = ntsb_inc[col].str.strip()

    ntsb_inc = process_code(ntsb_inc)
    ntsb_inc[' Airport Name '] = ntsb_inc[' Airport Name '].str.replace("N/A", "")
    ntsb_inc = process_times(ntsb_inc, verbose=verbose)

    print_preliminary_stats(ntsb_inc, verbose=verbose)

    # process airport name
    ntsb_inc['eventairport_conv'] = ntsb_inc[' Airport Name '].apply(process_ntsb_name)

    # process location cols
    ntsb_inc = process_location(ntsb_inc)

    assert(ntsb_inc[' Airport Code '].isna().sum() == 0 and \
            ntsb_inc[' Airport Name '].isna().sum() == 0)
    ntsb_inc = save_discarded(ntsb_inc)

    ntsb_inc = process_lat_lng(ntsb_inc)

    return ntsb_inc

def fill_in_handcode_iata(ntsb_inc):
    """
    This utilizes a handcoded dictionary that maps between airport name to matching IATA code.
    @param: ntsb_inc (pd.DataFrame) dataframe of NTSB incident/accident dataset
    @returns: processed ntsb_inc with handcoded IATA codes
    """
    fix_code_df = pd.read_excel('datasets/NTSB_airportcode_fix.xlsx', index_col=0)
    fix_code_dict = {}
    for index in fix_code_df.index:
        fix_code_dict[fix_code_df.loc[index, 'airport_code']] = index

    # clean airport code by removing first letter k if it has a length of 4
    len4_sel = ntsb_inc[' Airport Code '].str.len() == 4
    startswith_k = ntsb_inc[' Airport Code '].str.startswith('K')
    sel = len4_sel & startswith_k
    ntsb_inc.loc[sel, ' Airport code '] = ntsb_inc.loc[sel, ' Airport Code '].str.slice(1)

    ntsb_inc[' Airport Code '] = ntsb_inc[' Airport Code '].apply(lambda x: fix_code_dict.get(x, x))

    handcode_iata_dict_df = pd.read_excel('datasets/NTSB_Key.xlsx')
    handcode_iata_dict = {}
    for idx, row in handcode_iata_dict_df.set_index('airportname').iterrows():
        handcode_iata_dict[idx.strip()] = row['tracon_key']

    empty_code = ntsb_inc[' Airport Code '] == ''
    ntsb_inc.loc[empty_code, ' Airport Code '] = ntsb_inc[' Airport Name ']\
            .apply(lambda x: handcode_iata_dict.get(x, np.nan))

    codes = set(handcode_iata_dict.values())
    print('num covered by handcode', ntsb_inc[' Airport Code '].apply(lambda x: x in codes).sum())
    return ntsb_inc

def postprocess_results(ntsb_inc):
    """
    This post-processes the NTSB incident/accident dataset by grouping by airport code/year/month
    and calculates the number of incidents/accidents for that given tracon_month
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset after being processed
    """
    ntsb_inc = pd.concat([ntsb_inc, \
            pd.get_dummies(ntsb_inc[' Investigation Type '], prefix='ntsb')], axis=1)
    ntsb_inc = ntsb_inc[[' Airport Code ', 'year', 'month', 'ntsb_ Incident ', 'ntsb_ Accident ']]\
            .groupby([' Airport Code ', 'year', 'month']).sum()

    ntsb_inc = ntsb_inc.reset_index()
    ntsb_inc.rename({' Airport Code ': 'airport_code'}, axis=1, inplace=True)
    ntsb_inc.loc[ntsb_inc['airport_code'] == 'nan', 'airport_code'] = np.nan
    return ntsb_inc

def main(verbose=True):
    """
    Processes NTSB incident data
    """
    # preprocess data
    full = pd.read_csv('datasets/NTSB_AviationData_new.txt', sep="|")
    full = preprocess_data(full, verbose=verbose)

    full = fill_in_handcode_iata(full)

    # save dates
    tracon_date = pd.DataFrame(full.groupby(['day', 'month', 'year', ' Airport Code ']).size())
    tracon_date.to_csv('results/tracon_date_ntsb.csv')

    # postprocess results
    full = postprocess_results(full)
    full.to_csv('results/NTSB_AIDS_full_processed.csv', index=False)

if __name__ == "__main__":
    main()
