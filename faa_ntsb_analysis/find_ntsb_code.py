from collections import namedtuple
from tqdm import tqdm
from IPython import embed
from urllib.parse import quote
from urllib.error import HTTPError
from common_funcs import load_full_wiki, match_using_name_loc, search_wiki_airportname, \
        get_city, get_state, get_country
from selenium_funcs import check_code, search_city
import requests, pandas as pd, urllib.request as request, pickle
import re, numpy as np, pickle
page_google, search_wiki = False, False
match, create_backup = False, False
load_backupset = True
key = 'AIzaSyCR9FRYW-Y7JJbo4hU682rn6kJaUA5ABUc'
coverage = namedtuple('coverage', ['part', 'total'])

# load us state abbreviations
us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']

us_to_abbrev = {v: k for k,v in us_state_abbrev.items()}

def process_code(df):
    """
    Processes the Airport Code field in the NTSB incident/accident dataset by removing na values
    and turning everything to uppercase
    @param: df (pd.DataFrame) ntsb incident dataset
    @returns: df (pd.DataFrame) ntsb incident dataset w/filtered Airport Code
    """
    invalid_codes = ['n/a', 'none', 'na']
    for inv_code in invalid_codes:
        df[' Airport Code '] = df[' Airport Code '].str.replace(inv_code, "")
    nonna_code_sel = ~(df[' Airport Code '].isna())
    df.loc[nonna_code_sel, ' Airport Code '] = df.loc[nonna_code_sel, ' Airport Code '].str.upper()
    df.loc[df[' Airport Code '].isna(), ' Airport Code '] = ''
    return df

def process_times(df, verbose=True):
    """
    Processes the date in the NTSB incident/accident dataset.
    @param: df (pd.DataFrame) ntsb incident dataset
    @returns: df (pd.DataFrame) ntsb incident dataset w/processed date
    """
    df['year'] = df[' Event Date '].str.split("/").apply(lambda x: x[2]).apply(int)
    df['month'] = df[' Event Date '].str.split("/").apply(lambda x: x[0]).apply(int)
    df['day'] = df[' Event Date '].str.split("/").apply(lambda x: x[1]).apply(int)
    if verbose:
        print('full dataset size', df.shape[0])
    df = df.loc[(df['year'] >= 1988) & (df['year'] < 2020), :].copy()
    return df

def print_preliminary_stats(df, verbose=True):
    """
    Prints preliminary statistics on the NTSB incident/accident dataset.
    @param: df (pd.DataFrame) ntsb incident dataset
    """
    if verbose:
        print('airport code missing', df.loc[df[' Airport Code '] != ''].shape[0])
        print('airport name missing', df.loc[df[' Airport Name '] != ''].shape[0])
        print('airport code or name missing', df.loc[(df[' Airport Name '] != '') | \
                (df[' Airport Code '] != '')].shape[0])
        print('airport name and no code missing', df.loc[(df[' Airport Name '] != '') & \
                (df[' Airport Code '] == '')].shape[0])

def process_ntsb_name(x):
    """
    This processes the airport name field in the NTSB incident/accident dataset
    @param: x (str) name of the airport
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
        if word != "airport" and word != "-":
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

    x = str(x).lower()
    fin = []
    for elem in x.split(" "):
        process_word(fin, elem)
    return " ".join(fin)

def process_location(df):
    """
    This processes the location information in the NTSB incident/accident dataset
    @param: df (pd.DataFrame) ntsb incident dataset
    @returns: df (pd.DataFrame) ntsb incident dataset w/filtered location
    """
    df['eventcity'] = df[' Location '].apply(get_city)
    df['event_fullstate'] = df[' Location '].apply(get_state)
    df['event_country'] = df[' Location '].apply(get_country)
    return df

def save_discarded(df):
    """
    This saves the part of the NTSB incident/accident dataset that have no airport codes
    nor airport names to a separate csv file. Then, we utilize the rest of the dataset
    @param: df (pd.DataFrame) ntsb incident dataset
    @param: df (pd.DataFrame) ntsb incident dataset w/airport codes or names
    """
    code_and_name_empty = (df[' Airport Code '] == '') & (df[' Airport Name '] == '')
    df.loc[code_and_name_empty, :].to_csv('results/discarded_ntsb.csv')
    return df.loc[~code_and_name_empty, :].copy()

def conv_to_float(x):
    """
    Helper function that converts a string to a float (nan if it's an empty string)
    @param: x (str)
    @param: float version of x
    """
    if x == '':
        return np.nan
    else:
        return round(float(x.strip()), 5)

def process_lat_lng(df):
    """
    Processes the latitude/longitude fields in the NTSB incident/accident dataset.
    @param: df (pd.DataFrame) ntsb incident dataset
    @returns: df (pd.DataFrame) ntsb incident dataset w/processed latitude/longitude
    """
    df[' Latitude '] = df[' Latitude '].apply(conv_to_float)
    df[' Longitude '] = df[' Longitude '].apply(conv_to_float)
    return df

def preprocess_data(df, verbose=True):
    """
    Processes the NTSB incident/accident dataset by calling the above functions. It processes
    airport name, airport code, location, and latitude/longitude info.
    @param: df (pd.DataFrame) ntsb incident dataset
    @param: df (pd.DataFrame) processed ntsb incident dataset
    """
    strip_cols = [' Airport Code ' , ' Airport Name ', ' Latitude ', ' Longitude ']
    for col in strip_cols:
        df[col] = df[col].str.strip()

    df = process_code(df)
    df[' Airport Name '] = df[' Airport Name '].str.replace("N/A", "")
    df = process_times(df, verbose=verbose)

    print_preliminary_stats(df, verbose=verbose)

    # process airport name
    df['eventairport_conv'] = df[' Airport Name '].apply(process_ntsb_name)

    # process location cols
    df = process_location(df)

    assert(df[' Airport Code '].isna().sum() == 0 and df[' Airport Name '].isna().sum() == 0)
    df = save_discarded(df)

    df = process_lat_lng(df)

    return df

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

def save_nearest_airports(df):
    """
    This analyzes the part of the NTSB incident/accident dataset with empty airport codes, and
    non-empty latitude/longitude information. For these rows, we query world-airport-codes.com 
    to search for nearby airports, and save a dataframe that maps from latitude/longitude to
    nearby airports (as well as the estimated distance between the latitude/longitude coordinates
    and the nearby airports) -- saved to results/backup_ntsb.csv. A dictionary mapping from indices
    to errors (caused by querying the website) is also saved to (results/ntsb_backup_errors.pckl),
    and a set of queried indices is also saved (just in case the function fails in the middle).
    @param: df (pd.DataFrame) ntsb incident dataset
    """
    geo_cols = [' Latitude ', ' Longitude ', 'eventcity', 'event_country']
    empty_code_sel = df[' Airport Code '] == ''
    empty_lat, empty_lon = df[' Latitude '] == '', df[' Longitude '] == ''
    only_lat_lon = ~(empty_lat | empty_lon) & empty_code_sel

    if create_backup:
        backup_ntsb, ct, errors = [], 0, {}
        covered_set = set() # set of indices that we've already queried
        if load_backupset:
            covered_set = pickle.load(open('results/ntsb_backup_set.pckl', 'rb'))

        tmp_obj = df.loc[only_lat_lon, geo_cols].drop_duplicates()
        tqdm_obj = tqdm(tmp_obj.iterrows(), total = tmp_obj.shape[0], \
                desc = "wac near airports found 0")
        for idx, row in tqdm_obj:
            if idx in covered_set:
                continue
            try:
                res, found_ct = search_wac_row(df, row)
            except HTTPError as exception:
                errors[idx] = exception
            ct += found_ct

            if res is not None:
                backup_ntsb.append(res)
                tqdm_obj.set_description(f"wac near airports found {ct}")
            covered_set.add(idx)

            if len(covered_set) % 25 == 0:
                pickle.dump(covered_set, open('results/ntsb_backup_set.pckl', 'wb'))
                pickle.dump(list(errors.keys()), open('results/ntsb_backup_errors.pckl', 'wb'))
                pd.concat(backup_ntsb, axis = 0, ignore_index = True).to_csv('results/backup_ntsb.csv')

        pickle.dump(covered_set, open('results/ntsb_backup_set.pckl', 'wb'))
        pickle.dump(list(errors.keys()), open('results/ntsb_backup_errors.pckl', 'wb'))
        pd.concat(backup_ntsb, axis = 0, ignore_index = True).to_csv('results/backup_ntsb.csv')

def search_wikipedia(df):
    """
    This analyzes the part of the NTSB incident/accident dataset that has no airport code, but has
    an airport name. We query wikipedia utilizing the name and generate a dataframe that maps from
    airport name to IATA code (and other information scraped from wikipedia, see common_funcs.py for
    more info). We restrict our search space to only look at rows that take place within the US.
    @param: df (pd.DataFrame) ntsb incident dataset
    @returns: wiki_Search_found_pd (pd.DataFrame) dataframe that maps from airport name to IATA code
        as well as other information scraped from wikipedia (see common_funcs.py for more info)
    """
    name_nocode = (df[' Airport Code '] == '') & (df[' Airport Name '] != '')

    if search_wiki:
        wiki_search_found, total_ct = {}, 0
        name_and_state = df.loc[name_nocode, [' Airport Name ', 'event_fullstate', \
                'event_country']].drop_duplicates()
        tqdm_obj = tqdm(name_and_state.iterrows(), total = name_and_state.shape[0], \
                desc = "wiki found 0, tot 0")
        for idx, row in tqdm_obj:
            airportname, fullstate, country = row[' Airport Name '], row['event_fullstate'], row['event_country']
            if not pd.isna(airportname) and country == 'United States':
                res = search_wiki_airportname(airportname, fullstate)
                if res is not None:
                    wiki_search_found[airportname] = res
                    total_ct += df.loc[df[' Airport Name '] == airportname].shape[0]
                    tqdm_obj.set_description(f"wiki found {len(wiki_search_found)}, tot {total_ct}")
                    if len(wiki_search_found) % 25 == 0:
                        wiki_search_found_pd = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
                        wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found.csv')

        wiki_search_found_pd = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
        wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found.csv')
    else:
        wiki_search_found_pd = pd.read_csv('results/ntsb_wiki_search_found.csv', index_col = 0)
    return wiki_search_found_pd

def search_name_on_wiki_tables(df, verbose=True):
    """
    This function analyzes the part of the NTSB incident/accident dataset with no airport codes, but
    has an airport name. We scrape a set of dataframes from wikipedia.org (list of airport codes), and
    we search that dataframe utilizing the airport name. If the airport name matches with a row on the
    wikipedia table, then we fill in the airport code with the IATA code from wikipedia.org.
    @param: df (pd.DataFrame) ntsb incident dataset
    @param: verbose (bool) to print or not to print statistics
    @returns: df (pd.DataFrame) ntsb incident dataset with filled in IATA codes
    """
    wiki_tables = load_full_wiki(us_only = False)
    name_nocode = (df[' Airport Code '] == '') & (df[' Airport Name '] != '')
    res = match_using_name_loc(df[name_nocode], wiki_tables, col = ' Airport Name ')

    if verbose:
        print(df[df[' Airport Code '] == ''].shape)
    for idx, row in tqdm(res.iterrows(), total = res.shape[0]):
        df.loc[name_nocode & (df[' Airport Code '] == row['wiki_IATA']) & \
                (df['eventcity'] == row['eventcity']) 
                (df['event_fullstate'] == row['event_fullstate']), ' Airport Code '] = row['wiki_IATA']
    if verbose:
        print(df[df[' Airport Code '] == ''].shape)
    return df

def match_via_wikipedia(full, load_saved = True):
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
        selected_icao = wiki_tables.loc[wiki_tables['wiki_ICAO'] == code]
        return selected_iata.shape[0] > 0 or selected_iaco.shape[0] > 0

    # load + setup
    wiki_tables = load_full_wiki(us_only = False)
    matched_set, not_matched_set = set(), set()

    if load_saved:
        matched_set = pickle.load(open('results/ntsb_matched_set.pckl', 'rb'))
        not_matched_set = pickle.load(open('results/ntsb_not_matched_set.pckl', 'rb'))

    code_and_loc_pd =  full.loc[full[' Airport Code '] != '', [' Airport Code ']].drop_duplicates()
    tqdm_obj = tqdm(code_and_loc_pd.iterrows(), desc = f'match wiki found {len(matched_set)}', \
            total = code_and_loc_pd.shape[0])

    for idx, row in tqdm_obj:
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
    the given code, then we consider it a match. If there is a match, we add the code to the matched_set
    and otherwise, we add the code to the not_matched_set
    @param: full (pd.DataFrame) NTSB incident/accident dataset
    @param: matched_set (set of codes) generated from match_via_wikipedia of codes that have currently
        been matched 
    @param: not_matched_set (set of codes) generated from match_via_wikipedia of codes that have not
        been matched
    """
    code_and_loc_pd =  full.loc[full[' Airport Code '] != '', [' Airport Code ']].drop_duplicates()
    tqdm_obj = tqdm(code_and_loc_pd.iterrows(), total = code_and_loc_pd.shape[0],\
            desc = f'match wac found {len(matched_set)}')
    for idx, row in tqdm_obj:
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

def match_codes(df, verbose=True):
    """
    This creates a new column in the NTSB incident/accident dataset of whether or not the IATA code
    has been matched via a third-party website (either wikipedia or world-airport-codes.com). The 
    new column 'found_code_ntsb' indicates whether or not the airport code has been verified by another
    website
    @param: df (pd.DataFrame) ntsb incident dataset
    @param: verbose (bool) to print or not to print descriptive statistics
    @returns: df (pd.DataFrame) ntsb incident dataset w/added column
    """
    df['found_code_ntsb'] = 0
    if match:
        # see if the codes are matched in wikipedia table
        wiki_tables = load_full_wiki(us_only = False)
        matched_set, not_matched_set = match_via_wikipedia(df, load_saved = False)

        # count the number of codes matched in wikipedia table
        tmp = df.loc[~code_and_name_empty, :].copy()
        ct = 0
        for code in matched_set:
            ct += tmp.loc[tmp[' Airport Code '] == code].shape[0]
        if verbose:
            print('wiki matched', ct)

        match_via_wac(df, matched_set, not_matched_set)

        ct = 0
        for code in matched_set:
            sel = tmp[' Airport Code '] == code
            ct += tmp.loc[sel].shape[0]
        if verbose:
            print('wiki + airnav matched', ct)
        pickle.dump(matched_set, open('results/ntsb_matched_set.pckl', 'wb'))
    else:
        matched_set = pickle.load(open('results/ntsb_matched_set.pckl', 'rb'))

    df.loc[df[' Airport Code '].apply(lambda x: x in matched_set), 'found_code_ntsb'] = 1
    return df

def postprocess_results(df):
    """
    This post-processes the NTSB incident/accident dataset by grouping by airport code/year/month
    and calculates the number of incidents/accidents for that given tracon_month
    @param: df (pd.DataFrame) ntsb incident dataset
    @param: df (pd.DataFrame) ntsb incident dataset after being processed
    """
    df = pd.concat([df, pd.get_dummies(df[' Investigation Type '], prefix = 'ntsb')], axis = 1)
    df = df[[' Airport Code ', 'year', 'month', 'ntsb_ Incident ', 'ntsb_ Accident ']]\
            .groupby([' Airport Code ', 'year', 'month']).sum()

    df = df.reset_index()
    df.rename({' Airport Code ': 'airport_code'}, axis = 1, inplace = True)
    df.loc[df['airport_code'] == 'nan', 'airport_code'] = np.nan
    return df

def main(verbose=True):
    # preprocess data
    full = pd.read_csv('datasets/NTSB_AviationData_new.txt', sep = "|")
    full = preprocess_data(full, verbose=verbose)

    save_nearest_airports(full)

    # search wikipedia via requests
    wiki_search_found_pd = search_wikipedia(full)
    wiki_search_set = set(wiki_search_found_pd.index)
    if verbose:
        print(full.loc[full[' Airport Name '].apply(lambda x: x in wiki_search_set), :].shape)

    # search wikipedia on wiki tables
    full = search_name_on_wiki_tables(full, verbose=verbose)

    # match codes
    full = match_codes(full, verbose=verbose)

    # save dates
    tracon_date = pd.DataFrame(full.groupby(['day', 'month', 'year', ' Airport Code ']).size())
    tracon_date.to_csv('results/tracon_date_ntsb.csv')

    # postprocess results
    full = postprocess_results(full)
    full.to_csv('results/NTSB_AIDS_full_processed.csv', index = False)

if __name__ == "__main__":
    main()
