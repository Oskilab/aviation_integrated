from tqdm import tqdm
from IPython import embed
from common_funcs import load_full_wiki, match_using_name_loc, search_wiki_airportname
from requests import HTTPError
from selenium_funcs import check_code, search_city

import pandas as pd, numpy as np, re, pickle
import urllib.request as request
query_wiki, perform_name_matching = False, True
backup, check_codes = False, False
wac_load = True

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

    new.rename(rename_dict, axis = 1, inplace = True)
    return pd.concat([full, new], axis = 0, ignore_index = True, sort = False)

def get_month(date):
    """
    This extracts the month from the column 'localeventdate' in faa incident dataset.
    @param: date (str)
    @returns: month (int/float), nan if it cannot be extracted
    """
    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
                     'OCT', 'NOV', 'DEC']
    split_date = date.split("-")
    if len(split_date) != 3:
        return np.nan
    month_str = split_date[1]
    if month_str not in months:
        return np.nan
    else:
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
        if year >= 78 and year < 100:
            return year + 1900
        else:
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

def process_date(df):
    """
    This processes the dates of the faa incident dataset.
    @param: df (pd.DataFrame) faa incident dataset
    @returns: df (pd.DataFrame) faa incident dataset w/added time columns
    """
    df['month'] = df['localeventdate'].apply(get_month)
    df['year'] = df['localeventdate'].apply(get_year)
    df['day'] = df['localeventdate'].apply(get_day)
    df = df.loc[(df['year'] >= 1988) & (df['year'] < 2020)].copy()
    return df

def process_location(df):
    """
    This adds new columns to the dataframe regarding the location (state/city)
    @param: df (pd.DataFrame) faa incident dataset
    @returns: df (pd.DataFrame) faa incident dataset w/added location columns
    """
    # load us state abbreviations
    us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
    us_state_abbrev = us_state_abbrev.to_dict()['full']

    df['event_fullstate'] = df['eventstate'].apply(lambda x: us_state_abbrev.get(x, ''))
    df['eventcity'] = df['eventcity'].str.lower()
    return df

# needed for process_faa_name
replace = {'rgnl': ['regional'], 'fld':['field'], 'intl': ['international'], \
            'muni': ['municipal'], 'univ.' :['university'],\
            'iap': ['international', 'airport'], 'afb': ['air', 'force', 'base']}
def replace_with(s, replace = {}):
    """
    Helper function that replaces a word with its full form. The output however
    should be a list of words.
    @param: s (str) word
    @param: replace (dict: str->list of str) 
    @returns: list of replaced version of the word
    """
    return replace.get(s, [s])

def add_to_arr_replace(s, arr):
    """
    Add the replaced version of s to arr
    @param: s (str) word
    @param: arr (list) of final words
    """
    for replace_elem in replace_with(s, replace):
        arr.append(replace_elem)

def process_faa_name(x):
    """
    This process the name of an airport (cleaning extraneous characters, etc.)
    @param: x (str) name of an airport
    @returns: str of cleaned version of x
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

    x = str(x).lower()
    fin = []
    for word in x.split(" "):
        elem_arr = replace_with(word, replace)
        for elem in elem_arr:
            process_word(fin, elem)
    return " ".join(fin)

def combine_full_worldcities(full):
    """
    This adds latitude and longitude information about the city field in the faa incident
    dataset.
    @param: full (pd.DataFrame) faa incident dataset
    @returns: full (pd.DataFrame) with added columns 'city_lat', 'city_lon'
    """
    worldcities = pd.read_csv('datasets/worldcities.csv')
    set_of_worldcities = set(worldcities['city'].str.lower())

    full['city_lat'] = np.nan
    full['city_lon'] = np.nan
    for city in tqdm(full.loc[nan_airport_name_sel, 'eventcity'].unique(), \
            desc = "combining w/wordcities"):
        if city in set_of_worldcities:
            row_worldcities = worldcities[worldcities['city'].str.lower() == city].iloc[0]
            full.loc[full['eventcity'] == city, 'city_lat'] = row_worldcities['lat']
            full.loc[full['eventcity'] == city, 'city_lon'] = row_worldcities['lng']
    return full

def search_wac_nearby(full, nan_airport_name_sel, load=True):
    """
    This generates a dataframe of nearby airports utilizing latitude/longitude coordinates
    of the city provided in the faa incident dataset. This is accomplished by querying the
    world-airport-codes website (wac).
    @param: full (pd.DataFrame) faa incident dataset
    @param: nan_airport_name_sel (pd.Series of bool type) is a row selector. If the airport
        name is nan, then the corresponding element is True. It is used to select part
        of the full dataframe
    @param: load (bool) whether or not to load results from previous run
    @returns: pd.DataFrame of nearby airports (see selenium_funcs.py:search_city for more info)
    """
    def save_files():
        pickle.dump(results_dict, open('results/wac_search_results.pckl', 'wb'))
        pickle.dump(errors, open('results/wac_search_http_errors.pckl', 'wb'))
        pickle.dump(nf, open('results/wac_search_nf.pckl', 'wb'))
    geo_cols = ['eventcity', 'city_lat', 'city_lon']
    city_lat_lon = full.loc[nan_airport_name_sel, geo_cols].drop_duplicates()

    tqdm_obj = tqdm(city_lat_lon.iterrows(), total = city_lat_lon.shape[0], \
            desc = "search wac nearby found 0")
    results_dict, errors, nf = {}, {}, {}
    if load:
        results_dict = pickle.load(open('results/wac_search_results.pckl', 'rb'))
        errors = pickle.load(open('results/wac_search_http_errors.pckl', 'rb'))
        nf = pickle.load(open('results/wac_search_nf.pckl', 'rb'))

    for idx, row in tqdm_obj:
        if idx in results_dict or idx in errors or idx in nf:
            continue
        try:
            res = search_city(row['eventcity'], 'United States', row['city_lat'], row['city_lon'])
            if res is not None:
                results_dict[idx] = res
                tqdm_obj.set_description(f"search wac nearby found {len(results_dict)}")
                if len(results_dict) % 25 == 0:
                    save_files()
            else:
                nf[idx] = row['eventcity']
        except HTTPError as exception:
            errors[idx] = exception

    save_files()
    return pd.DataFrame.from_dict(results_dict, orient = 'index')

def create_backup(df, verbose=True):
    """
    This creates a backup dataframe of rows that have empty airport names. We try to find
    the latitude/longitude of the city associated with the row, and then find the closest
    airports to the center of the city. This is saved to a file (which we currently do 
    not use but may use in the future)
    @param: df (pd.DataFrame) faa incident dataset
    @param: verbose (bool) whether or not to print out statistics
    """
    nan_airport_name_sel = df['eventairport_conv'] == 'nan'
    if verbose:
        print('number of non_empty airport names', df.loc[~nan_airport_name_sel].shape[0])
    df = combine_full_worldcities(df) # add latitude longitude info to some cities

    nan_airport_name = df.loc[nan_airport_name_sel].copy()
    res = search_wac_nearby(df, nan_airport_name_sel, load = wac_search)

    backup_faa = []
    for idx in res.index:
        tmp = nan_airport_name.loc[nan_airport_name['eventcity'] == city_lat_lon.loc[idx, 'eventcity'],:].copy()
        for col in res.columns:
            tmp[col] = res.loc[idx, col]
        backup_faa.append(tmp)
    pd.concat(backup_faa, axis = 0, ignore_index = True).to_csv('results/backup_faa.csv')

def wiki_name_matching(df):
    """
    Given the faa incident dataset (which includes airport names), create a new dataframe that
    matches the airport names in the original df with a corresponding IATA code found in the
    wikipedia article of list of airport codes (utilizes common_funcs.py:load_full_wiki(...),
    and common_funcs.py:match_using_name_loc(...))
    @param: df (pd.DataFrame) faa incident dataset
    @returns: pd.DataFrame of airport names, and their corresponding IATA code on wikipedia.
        See common_funcs.py:match_using_name_loc(...) for more info.
    """
    if perform_name_matching:
        # load wikipedia airport name
        us_wiki_tables = load_full_wiki(us_only = True) # see common_funcs.py
        full_matched_pd = match_using_name_loc(df, us_wiki_tables)
        full_matched_pd.to_csv('matched_using_name.csv')
    else:
        full_matched_pd = pd.read_csv('matched_using_name.csv', index_col = 0)
    return full_matched_pd

def query_wikipedia(not_matched):
    """
    Given the dataframe created from get_unmatched_names_cities(...), or the dataset with
    airport names, cities and the number of times it occurs in the faa incident dataset, create
    a dataframe consisting of all the rows that can be connected to an article on wikipedia.

    @param: not_matched (pd.DataFrame) dataframe of unmatched airport names, its corresponding
        city, and the number of times they occur within the faa incident dataset
        result of get_unmatched_names_cities(...)
    @returns: pd.DataFrame of matched airport names (found by querying wikipedia).
        See common_funcs.py:search_wiki_airportname(...) for more info on the columns
    """
    if query_wiki:
        wiki_search = "https://en.wikipedia.org/w/index.php?search="
        wiki_search_found = {}

        name_state = not_matched[['eventairport_conv', 'event_fullstate']].drop_duplicates()
        tqdm_obj = tqdm(name_state.iterrows(), desc = "query wiki names found 0", total = name_state.shape[0])
        for idx, row in tqdm_obj:
            airportname, fullstate = row['eventairport_conv'], row['event_fullstate']
            res = search_wiki_airportname(airportname, fullstate)
            if res is not None:
                wiki_search_found[airportname] = res
                tqdm_obj.set_description(f"query wiki names found {len(wiki_search_found)}")

        wiki_search_found_df = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
        wiki_search_found_df.to_csv('results/wiki_search_found.csv')
    else:
        wiki_search_found_df = pd.read_csv('results/wiki_search_found.csv', index_col = 0)
    return wiki_search_found_df

def get_unmatched_names_cities(df, matched_names):
    """
    This creates a dataframe with airport name and its corresponding city (as well as how often
    it occurs within the faa incident dataset).
    @param: df (pd.DataFrame) faa incident dataset
    @param: matched_names (set of str) contains all airport names that have ben matched
        only via matching the airport names with the wikipedia table
    @returns: airport name/city/count pd.DataFrame
    """
    not_matched_sel = df['eventairport_conv'].apply(lambda x: x not in matched_names)
    return df.loc[not_matched_sel, ['eventairport_conv', 'eventcity']] \
            .groupby('eventairport_conv').count() \
            .sort_values(by = 'eventcity', ascending = False).reset_index()

def save_not_matched(df, not_matched, matched_names, verbose=True):
    """
    This function saves the portion of the faa incident dataset with airport names that have not
    been matched to an IATA code.
    @param: df (pd.DataFrame) faa incident dataset
    @param: not_matched (pd.DataFrame) contains the airport names and their corresponding cities
        and the number of times they occur in the faa incident dataset 
        (see get_unmatched_names_cities(...))
    @param: matched_names (set of str) contains all airport names that have been matched
    @param: verbose (bool) whether or not to print out statistics
    """
    if verbose:
        print(coverage(df.loc[df['eventairport_conv'].apply(lambda x: x in matched_names), :].shape[0],
                df.shape[0]))
    nf_sel = not_matched['eventairport_conv'].apply(lambda x: x not in matched_names)
    not_matched.loc[nf_sel, :].to_csv('results/not_matched.csv', index = False)

def fill_in_handcoded(df):
    """
    This fills in the faa incident dataset with IATA codes by utilizing the name of airport
    city and state, and a handcoded dataset with airport names/city/states and their corresponding
    airport codes
    @param: df (pd.DataFrame) faa incident dataset
    @returns: df (pd.DataFrame) with filled in iata codes
    """
    handcoded = pd.read_csv('datasets/not_matched_full_v1.csv', index_col = 0)
    handcoded.drop(['Unnamed: 8'], axis = 1, inplace = True)
    handcoded.rename({'Unnamed: 7': 'tracon_code'}, axis = 1,inplace = True)
    handcoded = handcoded.loc[~handcoded['tracon_code'].isna(), :].copy()

    for idx, row in tqdm(handcoded.iterrows(), total = handcoded.shape[0], desc = "handcoded"):
        if row['tracon_code'] != '?':
            df.loc[(df['eventairport_conv'] == row['eventairport_conv']) & \
                     (df['eventcity'] == row['eventcity']) & \
                     (df['eventstate'] == row['eventstate']), 'tracon_code'] = row['tracon_code']
    return df

def fill_in_iata(df, full_matched_pd, wiki_search_found_df):
    """
    Given the dataframe of matched wikipedia tracon_codes and a dataframe of matched tracon_codes
    from querying wikipedia (the latter is made via requests to wikipedia, the former is scraped),
    fill in the faa incident dataset with the corresponding tracon_code.
    @param: df (pd.DataFrame) faa incident dataset
    @param: full_matched_pd (pd.DataFrame) dataframe of matched airport names (from df), 
        and their corresponding airport codes found on a wikipedia table of airport codes
        see wiki_name_matching(...) for more info
    @param: wiki_search_found_df (pd.DataFrame) dataframe of matched airport names (from df),
        and their corresponding airport codes found by querying the wikipedia website
        see query_wikipedia(...) for more info
    @returns: df (pd.DataFrame) faa incident dataset with the tracon codes filled in
    """
    name_to_iata = {}
    for idx, row in full_matched_pd.iterrows():
        name_to_iata[row['eventairport_conv']] = row['wiki_IATA']
    for idx, row in wiki_search_found_df.iterrows():
        name_to_iata[idx] = row['iata']

    df['tracon_code'] = np.nan
    df['tracon_code'] = df['eventairport_conv'].apply(lambda x: name_to_iata[x] \
            if x in name_to_iata else np.nan)
    df = fill_in_handcoded(df)
    return df

def wiki_matched_set(df, wiki_tables):
    """
    This creates a set of tracon_codes that are found in the faa incident dataset and
    the wikipedia table of tracon_codes.
    @param: df (pd.DataFrame) faa incident dataset
    @param: wiki_tables (pd.DataFrame) wikipedia airport dataset (see common_funcs:load_full_wiki)
    """
    matched_set = set()
    code_and_loc_pd = df.loc[~df['tracon_code'].isna(), ['tracon_code']].drop_duplicates()
    for idx, row in code_and_loc_pd.iterrows():
        selected = wiki_tables.loc[wiki_tables['wiki_IATA'] == row['tracon_code']]
        if selected.shape[0] > 0:
            matched_set.add(row['tracon_code'])
    return matched_set

def count_matched(df, matched_set):
    """
    This counts the number of rows with a tracon code within a given set.
    @param: df (pd.DataFrame) faa incident dataset
    @param: matched_set (set of tracon codes)
    @returns: ct (int) number of rows with a tracon code within the given set
    """
    ct = 0
    for code in matched_set:
        sel = df['tracon_code'] == code
        ct += df.loc[sel].shape[0]
    return ct

def check_tracon_codes(df):
    """
    Double check that codes in df are in wikipedia or airnav, and adds a new column that
    indicates whether or not the code was found in wikipedia/airnav.
    @param: df (pd.DataFrame) of faa incidents
    @returns: df (pd.DataFrame) with added column
    """
    df['found_code_faa'] = 0
    if check_codes:
        # match with wiki
        wiki_tables = load_full_wiki(us_only = False)
        matched_set = wiki_matched_set(df, wiki_tables)

        tmp = df.loc[df['eventairport_conv'] != 'nan', :].copy()
        ct = count_matched(tmp, matched_set)
        print('wiki matched', ct)

        # match with wac
        for idx, row in tqdm(code_and_loc_pd.iterrows(), total = code_and_loc_pd.shape[0]):
            if row['tracon_code'] not in matched_set and check_code(row['tracon_code']):
                matched_set.add(row['tracon_code'])

        print('wiki + airnav matched', count_matched(tmp, matched_set))
        pickle.dump(matched_set, open('results/matched_set_faa.pckl', 'wb'))
    else:
        if False:
            matched_set = pickle.load(open('results/matched_set_faa.pckl', 'rb'))
    # TODO: you need to rerun this while also checking codes. I neglected to add this to the
    # github so this file is lost. Takes a while to run.
    # df.loc[df['tracon_code'].apply(lambda x: x in matched_set), 'found_code_faa'] = 1
    return df

def post_process_results(df):
    """
    This post processes the results by grouping by tracon_month and then counting the number
    of rows (and setting a new column faa_incidents equal to the number of rows). Also deals
    with nans/none.
    @param: df (pd.DataFrame) dataframe at the end of the pipeline
    @returns: processed df (pd.DataFrame)
    """
    # hack around groupby ignoring nan values
    df['tracon_code'] = df['tracon_code'].fillna('nan') 

    tracon_date = pd.DataFrame(df.groupby(['day', 'month', 'year', 'tracon_code']).size())
    tracon_date.to_csv('results/tracon_date_faa.csv')

    cols = ['tracon_code', 'month', 'year', 'eventtype']
    df = df[cols].groupby(cols[:-1]).count().rename({cols[-1]: 'faa_incidents'}, axis = 1).reset_index()
    df['tracon_code'] = df['tracon_code'].str.replace("none", "")

    # deal with na values
    df.loc[df['tracon_code'] == 'nan', 'tracon_code'] = np.nan
    df.loc[df['tracon_code'] == 'none', 'tracon_code'] = np.nan
    return df

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

    # creates backup faa file. If the airport name is nan, then we look at city names. We 
    # combine this with a list of downloaded worldcities (and their corresponding lat/lng) to
    # create a list of potential airports
    if backup:
        create_backup(full, verbose=verbose)

    # match names using wikipedia table
    full_matched_pd = wiki_name_matching(full)

    # work on those that we could not find codes for
    all_matched_names = set(full_matched_pd['eventairport_conv'])
    not_matched = get_unmatched_names_cities(full, all_matched_names)

    # match names by querying wikipedia
    wiki_search_found_df = query_wikipedia(not_matched)

    # save unmatched part of dataset to separate csv file
    all_matched_names = all_matched_names.union(set(wiki_search_found_df.index))
    save_not_matched(full, not_matched, all_matched_names, verbose=verbose)

    # fill in dataset with matched iata codes (found via airport name above)
    full = fill_in_iata(full, full_matched_pd, wiki_search_found_df)
    if verbose:
        print('non empty tracon code rows', full.loc[~full['tracon_code'].isna()].shape[0])

    full = check_tracon_codes(full)

    full = post_process_results(full)
    full.to_csv('results/FAA_AIDS_full_processed.csv', index = False)

if __name__ == "__main__":
    main()
