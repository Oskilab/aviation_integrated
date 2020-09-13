from tqdm import tqdm
from urllib.parse import quote_plus, quote
from collections import namedtuple
from bs4 import BeautifulSoup
from IPython import embed
from common_funcs import load_full_wiki, match_using_name_loc, search_wiki_airportname
from requests import HTTPError
from selenium_funcs import check_code, search_city

import pandas as pd, numpy as np, re, ssl, pickle, os
import urllib.request as request
coverage = namedtuple('coverage', ['part', 'total'])
query_wiki, perform_name_matching = False, True
create_backup, check_codes = False, False
wac_load = True

# if check_codes and not os.path.exists('results/matched_set_faa.pckl'):
#     raise Exception("if check code is set to True, then 'results/matched_set_faa.pckl' must exist")
    

def load_faa_data():
    full = pd.read_csv('datasets/FAA_AIDS_full.csv')
    new = pd.read_csv('datasets/FAA_AIDS_addition.csv')

    # rename columns
    rename_dict = {}
    for col in new.columns:
        rename_dict[col] = col.lower().replace(" ", "")

    new.rename(rename_dict, axis = 1, inplace = True)
    return pd.concat([full, new], axis = 0, ignore_index = True, sort = False)

def get_month(date):
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
    split_date = date.split("-")
    if len(split_date) != 3:
        return np.nan
    day_str = split_date[0]
    try:
        day = int(day_str)
        return day
    except ValueError:
        return np.nan

# load datasets
full = load_faa_data()
print('total original number of rows', full.shape[0])

# filter on date
full['month'] = full['localeventdate'].apply(get_month)
full['year'] = full['localeventdate'].apply(get_year)
full['day'] = full['localeventdate'].apply(get_day)
full = full.loc[(full['year'] >= 1988) & (full['year'] < 2020)].copy()
print('number of rows, filtered by date', full.shape[0])

# load us state abbreviations
us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']

"""
We need to process FAA airport name in order to utilize it for wikipedia tracon
searches.
"""

# needed for process_faa_name
reg_dash = '([a-z]{1,})[-/]{1}([a-z]{1,})'
pat = re.compile(reg_dash)
replace = {'rgnl': ['regional'], 'fld':['field'], 'intl': ['international'], \
            'muni': ['municipal'], 'univ.' :['university'],\
            'iap': ['international', 'airport'], 'afb': ['air', 'force', 'base']}
def replace_with(s, replace = {}):
    return replace.get(s, [s])

def add_to_arr_replace(s, arr):
    for replace_elem in replace_with(s, replace):
        arr.append(replace_elem)

def process_faa_name(x):
    x = str(x).lower()
    fin = []
    for elem in x.split(" "):
        elem_arr = replace_with(elem, replace)
        for elem in elem_arr:
            if elem != "-":
                if elem.startswith("(") or elem.startswith("/"):
                    elem = elem[1:]
                if elem.endswith(")") or elem.endswith("/") or elem.endswith(",") or elem.endswith("-"):
                    elem = elem[:-1]
                # add to fin
                reg_res = pat.match(elem)
                if reg_res is not None:
                    add_to_arr_replace(reg_res.group(1), fin)
                    add_to_arr_replace(reg_res.group(2), fin)
                else:
                    add_to_arr_replace(elem, fin)
    return " ".join(fin)

full['event_fullstate'] = full['eventstate'].apply(lambda x: us_state_abbrev.get(x, ''))
full['eventcity'] = full['eventcity'].str.lower()
full['eventairport_conv'] = full['eventairport'].apply(process_faa_name)

"""
Create backup FAA file. If the airportname is nan, then we look at city names. We combine this
with a list of downloaded worldcities (and their corresponding latitude/longitude), to create
a list of potential airports
"""

def combine_full_worldcities(full):
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

def search_wac_nearby(full, load = True):
    def save_files():
        pickle.dump(results_dict, open('results/wac_search_results.pckl', 'wb'))
        pickle.dump(errors, open('results/wac_search_http_errors.pckl', 'wb'))
        pickle.dump(errors, open('results/wac_search_nf.pckl', 'wb'))
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

nan_airport_name_sel = full['eventairport_conv'] == 'nan'
print('number of non_empty airport names', full.loc[~nan_airport_name_sel].shape[0])

if create_backup:
    full = combine_full_wordcities(full) # add latitude longitude info to some cities

    nan_airport_name = full.loc[nan_airport_name_sel].copy()
    res = search_wac_nearby(full, load = wac_search)

    backup_faa = []
    for idx in res.index:
        tmp = nan_airport_name.loc[nan_airport_name['eventcity'] == city_lat_lon.loc[idx, 'eventcity'],:].copy()
        for col in res.columns:
            tmp[col] = res.loc[idx, col]
        backup_faa.append(tmp)
    pd.concat(backup_faa, axis = 0, ignore_index = True).to_csv('results/backup_faa.csv')


"""
Match names using wikipedia table.
"""

if perform_name_matching:
    # load wikipedia airport name
    us_wiki_tables = load_full_wiki(us_only = True) # see common_funcs.py
    full_matched_pd = match_using_name_loc(full,us_wiki_tables)
    full_matched_pd.to_csv('matched_using_name.csv')
else:
    full_matched_pd = pd.read_csv('matched_using_name.csv', index_col = 0)

all_matched_names = set(full_matched_pd['eventairport_conv'])

"""
Match names querying wikipedia.
"""
# work on those that we could not find codes for
not_matched_sel = full['eventairport_conv'].apply(lambda x: x not in all_matched_names)
not_matched = full.loc[not_matched_sel, ['eventairport_conv', 'eventcity']]\
        .groupby('eventairport_conv').count()\
        .sort_values(by = 'eventcity', ascending = False).reset_index()

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

wiki_search_found_set = set(wiki_search_found_df.index)

# all the names we've matched to codes
total_matched_names = all_matched_names.union(wiki_search_found_set)
print(coverage(full.loc[full['eventairport_conv'].apply(lambda x: x in total_matched_names), :].shape[0],
        full.shape[0]))

"""
Fill in dataset with matched iata codes (found via airport name above)
"""
nf_sel = not_matched['eventairport_conv'].apply(lambda x: x not in total_matched_names)
not_matched.loc[nf_sel, :].to_csv('results/not_matched.csv', index = False)

name_to_iata = {}
for idx, row in full_matched_pd.iterrows():
    name_to_iata[row['eventairport_conv']] = row['wiki_IATA']
for idx, row in wiki_search_found_df.iterrows():
    name_to_iata[idx] = row['iata']

full['tracon_code'] = np.nan
full['tracon_code'] = full['eventairport_conv'].apply(lambda x: name_to_iata[x] if x in name_to_iata else np.nan)

# then perform hand-coded analysis
handcoded = pd.read_csv('datasets/not_matched_full_v1.csv', index_col = 0)
handcoded.drop(['Unnamed: 8'], axis = 1, inplace = True)
handcoded.rename({'Unnamed: 7': 'tracon_code'}, axis = 1,inplace = True)
handcoded = handcoded.loc[~handcoded['tracon_code'].isna(), :].copy()

for idx, row in tqdm(handcoded.iterrows(), total = handcoded.shape[0], desc = "handcoded"):
    if row['tracon_code'] != '?':
        full.loc[(full['eventairport_conv'] == row['eventairport_conv']) & \
                 (full['eventcity'] == row['eventcity']) & \
                 (full['eventstate'] == row['eventstate']), 'tracon_code'] = row['tracon_code']

print('non empty tracon code rows', full.loc[~full['tracon_code'].isna()].shape[0])

"""
Double check that codes are in wikipedia or airnav
"""
full['found_code_faa'] = 0
if check_codes:
    # match with wiki
    wiki_tables = load_full_wiki(us_only = False)
    matched_set = set()

    code_and_loc_pd =  full.loc[~full['tracon_code'].isna(), ['tracon_code']].drop_duplicates()
    for idx, row in code_and_loc_pd.iterrows():
        selected = wiki_tables.loc[wiki_tables['wiki_IATA'] == row['tracon_code']]
        if selected.shape[0] > 0:
            matched_set.add(row['tracon_code'])

    tmp = full.loc[full['eventairport_conv'] != 'nan', :].copy()
    ct = 0
    for code in matched_set:
        sel = tmp['tracon_code'] == code
        ct += tmp.loc[sel].shape[0]
    print('wiki matched', ct)

    # match with wac
    for idx, row in tqdm(code_and_loc_pd.iterrows(), total = code_and_loc_pd.shape[0]):
        if row['tracon_code'] not in matched_set:
            if check_code(row['tracon_code']):
                matched_set.add(row['tracon_code'])
    ct = 0
    for code in matched_set:
        sel = tmp['tracon_code'] == code
        ct += tmp.loc[sel].shape[0]
    print('wiki + airnav matched', ct)
    pickle.dump(matched_set, open('results/matched_set_faa.pckl', 'wb'))
else:
    matched_set = pickle.load(open('results/matched_set_faa.pckl', 'rb'))

full.loc[full['tracon_code'].apply(lambda x: x in matched_set), 'found_code_faa'] = 1

"""
Get results
"""

# hack around groupby ignoring nan values
full['tracon_code'] = full['tracon_code'].fillna('nan') 

# vol_tracons = set(pickle.load(open('../results/vol_data.pckl', 'rb')))
# num_na = (full['tracon_code'] == 'nan').sum()
# print('vol match', coverage(full['tracon_code'].apply(lambda x: x in vol_tracons).sum(), \
#         full.shape[0]), f'na codes {num_na}')
tracon_date = pd.DataFrame(full.groupby(['day', 'month', 'year', 'tracon_code']).size())
tracon_date.to_csv('results/tracon_date_faa.csv')

cols = ['tracon_code', 'month', 'year', 'eventtype']
full = full[cols].groupby(cols[:-1]).count().rename({cols[-1]: 'faa_incidents'}, axis = 1).reset_index()
full['tracon_code'] = full['tracon_code'].str.replace("none", "")

# deal with na values
full.loc[full['tracon_code'] == 'nan', 'tracon_code'] = np.nan
full.loc[full['tracon_code'] == 'none', 'tracon_code'] = np.nan

full.to_csv('results/FAA_AIDS_full_processed.csv', index = False)
