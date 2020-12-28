"""
Loads NTSB incident/accident dataset and creates tracon_code column
"""
import re
import pickle
from urllib.error import HTTPError

from tqdm import tqdm
import numpy as np
import pandas as pd

from selenium_funcs import check_code, search_city

from common_funcs import load_full_wiki, match_using_name_loc, search_wiki_airportname, \
        get_city, get_state, get_country

page_google, search_wiki = False, False
match, create_backup = False, False
LOAD_BACKUPSET = True

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

def search_wac_row(full, row):
    """
    This utilizes the row parameter to search for nearby airports utilizing its latitude and
    longitude information, city, and country. If some airports are found, then this function
    selects the part of full (the NTSB incident/accident dataset) that have that latitude
    and longitude info and returns it (as well as the number of rows within that dataset).
    @param: full (pd.DataFrame) ntsb incident dataset
    @param: row (pd.Series) row of latitude/longitude/city/country information.
        This should be extracted from the NTSB incident/accident dataset.
        Must have the following columns 'eventcity', 'event_country', ' Latitude ', and
        ' Longitude '
    @returns: pd.DataFrame. Subset of the input parameter full that contains the latitude and
        longitude information that we queried to world-airport-codes.com
    @returns: num_rows (int), the number of rows in our output dataframe
    """
    lat, lon = row[' Latitude '], row[' Longitude ']

    # the following call returns a table of airports that are nearest the given latitude and
    # longitude information given by the parameter row. This is done by querying
    # world-airport-codes.com
    res = search_city(row['eventcity'], row['event_country'], \
            row[' Latitude '], row[' Longitude '])

    if res is not None:
        lat_sel = full[' Latitude '].apply(lambda x: round(x, 5)) == lat
        lon_sel = full[' Longitude '].apply(lambda x: round(x, 5)) == lon

        tmp = full.loc[lat_sel & lon_sel, :].copy()

        # add new columns
        for col in res.index:
            tmp[col] = res.loc[col]

        return tmp, tmp.shape[0]
    return None, 0

def save_nearest_airports(ntsb_inc):
    """
    This analyzes the part of the NTSB incident/accident dataset with empty airport codes, and
    non-empty latitude/longitude information. For these rows, we query world-airport-codes.com
    to search for nearby airports, and save a dataframe that maps from latitude/longitude to
    nearby airports (as well as the estimated distance between the latitude/longitude coordinates
    and the nearby airports) -- saved to results/backup_ntsb.csv. A dictionary mapping from indices
    to errors (caused by querying the website) is also saved to (results/ntsb_backup_errors.pckl),
    and a set of queried indices is also saved (just in case the function fails in the middle).
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    """
    geo_cols = [' Latitude ', ' Longitude ', 'eventcity', 'event_country']
    empty_code_sel = ntsb_inc[' Airport Code '] == ''
    empty_lat, empty_lon = ntsb_inc[' Latitude '] == '', ntsb_inc[' Longitude '] == ''
    only_lat_lon = ~(empty_lat | empty_lon) & empty_code_sel

    if create_backup:
        backup_ntsb, ctr, errors = [], 0, {}
        covered_set = set() # set of indices that we've already queried
        if LOAD_BACKUPSET:
            covered_set = pickle.load(open('results/ntsb_backup_set.pckl', 'rb'))

        tmp_obj = ntsb_inc.loc[only_lat_lon, geo_cols].drop_duplicates()
        tqdm_obj = tqdm(tmp_obj.iterrows(), total=tmp_obj.shape[0], \
                desc="wac near airports found 0")
        for idx, row in tqdm_obj:
            if idx in covered_set:
                continue
            try:
                res, found_ct = search_wac_row(ntsb_inc, row)
            except HTTPError as exception:
                errors[idx] = exception
            ctr += found_ct

            if res is not None:
                backup_ntsb.append(res)
                tqdm_obj.set_description(f"wac near airports found {ctr}")
            covered_set.add(idx)

            if len(covered_set) % 25 == 0:
                pickle.dump(covered_set, open('results/ntsb_backup_set.pckl', 'wb'))
                pickle.dump(list(errors.keys()), open('results/ntsb_backup_errors.pckl', 'wb'))
                pd.concat(backup_ntsb, axis=0, ignore_index=True).to_csv('results/backup_ntsb.csv')

        pickle.dump(covered_set, open('results/ntsb_backup_set.pckl', 'wb'))
        pickle.dump(list(errors.keys()), open('results/ntsb_backup_errors.pckl', 'wb'))
        pd.concat(backup_ntsb, axis=0, ignore_index=True).to_csv('results/backup_ntsb.csv')

def search_wikipedia(ntsb_inc):
    """
    This analyzes the part of the NTSB incident/accident dataset that has no airport code, but has
    an airport name. We query wikipedia utilizing the name and generate a dataframe that maps from
    airport name to IATA code (and other information scraped from wikipedia, see common_funcs.py for
    more info). We restrict our search space to only look at rows that take place within the US.
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @returns: wiki_Search_found_pd (pd.DataFrame) dataframe that maps from airport name to IATA code
        as well as other information scraped from wikipedia (see common_funcs.py for more info)
    """
    name_nocode = (ntsb_inc[' Airport Code '] == '') & (ntsb_inc[' Airport Name '] != '')

    if search_wiki:
        wiki_search_found, total_ct = {}, 0
        name_and_state = ntsb_inc.loc[name_nocode, [' Airport Name ', 'event_fullstate', \
                'event_country']].drop_duplicates()
        tqdm_obj = tqdm(name_and_state.iterrows(), total=name_and_state.shape[0], \
                desc="wiki found 0, tot 0")
        for _, row in tqdm_obj:
            airportname, fullstate, country = \
                    row[' Airport Name '], row['event_fullstate'], row['event_country']
            if not pd.isna(airportname) and country == 'United States':
                res = search_wiki_airportname(airportname, fullstate)
                if res is not None:
                    wiki_search_found[airportname] = res
                    total_ct += ntsb_inc.loc[ntsb_inc[' Airport Name '] == airportname].shape[0]
                    tqdm_obj.set_description(f"wiki found {len(wiki_search_found)}, tot {total_ct}")
                    if len(wiki_search_found) % 25 == 0:
                        wiki_search_found_pd = \
                                pd.DataFrame.from_dict(wiki_search_found, orient='index')
                        wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found.csv')

        wiki_search_found_pd = pd.DataFrame.from_dict(wiki_search_found, orient='index')
        wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found.csv')
    else:
        wiki_search_found_pd = pd.read_csv('results/ntsb_wiki_search_found.csv', index_col=0)
    return wiki_search_found_pd

def search_name_on_wiki_tables(ntsb_inc, verbose=True):
    """
    This function analyzes the part of the NTSB incident/accident dataset with no airport codes, but
    has an airport name. We scrape a set of dataframes from wikipedia.org (list of airport codes),
    and we search that dataframe utilizing the airport name. If the airport name matches with a
    row on  the wikipedia table, then we fill in the airport code with the IATA code from
    wikipedia.org.
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @param: verbose (bool) to print or not to print statistics
    @returns: ntsb_inc (pd.DataFrame) ntsb incident dataset with filled in IATA codes
    """
    wiki_tables = load_full_wiki(us_only=False)
    name_nocode = (ntsb_inc[' Airport Code '] == '') & (ntsb_inc[' Airport Name '] != '')
    res = match_using_name_loc(ntsb_inc[name_nocode], wiki_tables, col=' Airport Name ')

    if verbose:
        print(ntsb_inc[ntsb_inc[' Airport Code '] == ''].shape)
    for _, row in tqdm(res.iterrows(), total=res.shape[0]):
        sel = name_nocode & (ntsb_inc[' Airport Code '] == row['wiki_IATA']) & \
                (ntsb_inc['eventcity'] == row['eventcity']) & \
                (ntsb_inc['event_fullstate'] == row['event_fullstate'])

        ntsb_inc.loc[sel, ' Airport Code '] = row['wiki_IATA']
    if verbose:
        print(ntsb_inc[ntsb_inc[' Airport Code '] == ''].shape)
    return ntsb_inc

def match_via_wikipedia(full, load_saved=True):
    """
    We are attempting to match airport codes found in the NTSB incident/accident dataset to
    airport codes found from the wikipedia tables. This function generates a set of airport codes
    found in the NTSB incident/accident dataset that also match with an airport code in the
    wikipedia dataset (matched_set), and a set of airport codes found in the NTSB incident/accident
    dataset but not in the wikipedia dataset (not_matched_set)
    @param: full (pd.DataFrame) NTSB incident/accident dataset
    @param: load_saved (bool) whether or not to load the matched_set and unmatched set that we
        generated in a previous running of this script
    @returns: matched_set (set of str/codes), not_matched_set (set of str/codes)
    """
    def match_row(code):
        """
        Returns whether or not there is a match in the wikipedia dataset.
        @param: code (str)
        @returns: bool of whether or not there is match
        """
        selected_iata = wiki_tables.loc[wiki_tables['wiki_IATA'] == code]
        selected_iaco = wiki_tables.loc[wiki_tables['wiki_ICAO'] == code]
        return selected_iata.shape[0] > 0 or selected_iaco.shape[0] > 0

    # load + setup
    wiki_tables = load_full_wiki(us_only=False)
    matched_set, not_matched_set = set(), set()

    if load_saved:
        matched_set = pickle.load(open('results/ntsb_matched_set.pckl', 'rb'))
        not_matched_set = pickle.load(open('results/ntsb_not_matched_set.pckl', 'rb'))

    code_and_loc_pd = full.loc[full[' Airport Code '] != '', [' Airport Code ']].drop_duplicates()
    tqdm_obj = tqdm(code_and_loc_pd.iterrows(), desc=f'match wiki found {len(matched_set)}', \
            total=code_and_loc_pd.shape[0])

    for _, row in tqdm_obj:
        code = row[' Airport Code ']
        if code in matched_set:
            continue
        if match_row(code):
            matched_set.add(code)
            if len(matched_set) % 25 == 0:
                pickle.dump(matched_set, open('results/ntsb_matched_set.pckl', 'wb'))
                pickle.dump(not_matched_set, open('results/ntsb_not_matched_set.pckl', 'wb'))

            tqdm_obj.set_description(f'found {len(matched_set)}')
        else:
            not_matched_set.add(code)
    return matched_set, not_matched_set

def match_via_wac(full, matched_set, not_matched_set):
    """
    This function matches airport codes found in the NTSB incident/accident dataset to airport codes
    found by querying world-airport-codes.com. If there is an entry in world-airport-codes with
    the given code, then we consider it a match. If there is a match, we add the code to
    the matched_set and otherwise, we add the code to the not_matched_set
    @param: full (pd.DataFrame) NTSB incident/accident dataset
    @param: matched_set (set of codes) generated from match_via_wikipedia of codes that
        have currently been matched
    @param: not_matched_set (set of codes) generated from match_via_wikipedia of codes
        that have not been matched
    """
    code_and_loc_pd = full.loc[full[' Airport Code '] != '', [' Airport Code ']].drop_duplicates()
    tqdm_obj = tqdm(code_and_loc_pd.iterrows(), total=code_and_loc_pd.shape[0],\
            desc=f'match wac found {len(matched_set)}')
    for _, row in tqdm_obj:
        code = row[' Airport Code ']
        if code not in matched_set:
            if check_code(code):
                matched_set.add(code)
                tqdm_obj.set_description(f'match wac found {len(matched_set)}')
                if len(matched_set) % 25 == 0:
                    pickle.dump(matched_set, open('results/ntsb_matched_set.pckl', 'wb'))
                    pickle.dump(not_matched_set, open('results/ntsb_not_matched_set.pckl', 'wb'))
            else:
                not_matched_set.add(code)
        else:
            not_matched_set.add(code)

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

def match_codes(ntsb_inc, verbose=True):
    """
    This creates a new column in the NTSB incident/accident dataset of whether or not the IATA code
    has been matched via a third-party website (either wikipedia or world-airport-codes.com). The
    new column 'found_code_ntsb' indicates whether or not the airport code has been
    verified by another website
    @param: ntsb_inc (pd.DataFrame) ntsb incident dataset
    @param: verbose (bool) to print or not to print descriptive statistics
    @returns: ntsb_inc (pd.DataFrame) ntsb incident dataset w/added column
    """
    ntsb_inc['found_code_ntsb'] = 0
    if match:
        # see if the codes are matched in wikipedia table
        # wiki_tables = load_full_wiki(us_only=False)
        matched_set, not_matched_set = match_via_wikipedia(ntsb_inc, load_saved=False)

        code_and_name_empty = (ntsb_inc[' Airport Code '] == '') & \
                (ntsb_inc[' Airport Name '] == '')
        # count the number of codes matched in wikipedia table
        tmp = ntsb_inc.loc[~code_and_name_empty, :].copy()
        ctr = 0
        for code in matched_set:
            ctr += tmp.loc[tmp[' Airport Code '] == code].shape[0]
        if verbose:
            print('wiki matched', ctr)

        match_via_wac(ntsb_inc, matched_set, not_matched_set)

        ctr = 0
        for code in matched_set:
            sel = tmp[' Airport Code '] == code
            ctr += tmp.loc[sel].shape[0]
        if verbose:
            print('wiki + airnav matched', ctr)
        pickle.dump(matched_set, open('results/ntsb_matched_set.pckl', 'wb'))
    else:
        matched_set = pickle.load(open('results/ntsb_matched_set.pckl', 'rb'))

    ntsb_inc.loc[\
            ntsb_inc[' Airport Code '].apply(lambda x: x in matched_set), 'found_code_ntsb'] = 1
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

    save_nearest_airports(full)

    # the following lines are commented out because matching names to IATA codes created
    # some number of false positives. Instead, we now utilize a dictionary that maps from
    # eventairport to IATA code

    # search wikipedia via requests
    # wiki_search_found_pd = search_wikipedia(full)
    # wiki_search_set = set(wiki_search_found_pd.index)
    # if verbose:
    #     print(full.loc[full[' Airport Name '].apply(lambda x: x in wiki_search_set), :].shape)

    # search wikipedia on wiki tables
    # full = search_name_on_wiki_tables(full, verbose=verbose)

    full = fill_in_handcode_iata(full)

    # match codes
    full = match_codes(full, verbose=verbose)

    # save dates
    tracon_date = pd.DataFrame(full.groupby(['day', 'month', 'year', ' Airport Code ']).size())
    tracon_date.to_csv('results/tracon_date_ntsb.csv')

    # postprocess results
    full = postprocess_results(full)
    full.to_csv('results/NTSB_AIDS_full_processed.csv', index=False)

if __name__ == "__main__":
    main()
