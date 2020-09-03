from tqdm import tqdm
from urllib.parse import quote_plus, quote
from collections import namedtuple
from bs4 import BeautifulSoup
from IPython import embed
from common_funcs import load_full_wiki, match_using_name_loc, search_wiki_airportname
import pandas as pd, numpy as np, re, ssl, pickle
import urllib.request as request
from selenium_funcs import check_code
coverage = namedtuple('coverage', ['part', 'total'])
query_wiki, perform_name_matching = False, True

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
# load datasets
full = load_faa_data()
print('total original number of rows', full.shape[0])

# filter on date
full['month'] = full['localeventdate'].apply(get_month)
full['year'] = full['localeventdate'].apply(get_year)
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
embed()
1/0

nan_airport_name_sel = full['eventairport_conv'] == 'nan'
print('number of non_empty airport names', full.loc[~nan_airport_name_sel].shape[0])

if perform_name_matching:
    # load wikipedia airport name
    us_wiki_tables = load_full_wiki(us_only = True) # see common_funcs.py
    full_matched_pd = match_using_name_loc(full,us_wiki_tables)
    full_matched_pd.to_csv('matched_using_name.csv')
else:
    full_matched_pd = pd.read_csv('matched_using_name.csv', index_col = 0)

all_matched_names = set(full_matched_pd['eventairport_conv'])

# work on those that we could not find codes for
not_matched_sel = full['eventairport_conv'].apply(lambda x: x not in all_matched_names)
not_matched = full.loc[not_matched_sel, ['eventairport_conv', 'eventcity']]\
        .groupby('eventairport_conv').count()\
        .sort_values(by = 'eventcity', ascending = False).reset_index()

if query_wiki:
    wiki_search = "https://en.wikipedia.org/w/index.php?search="
    wiki_search_found = {}

    name_state = not_matched[['eventairport_conv', 'event_fullstate']].drop_duplicates()
    tqdm_obj = tqdm(name_state.iterrows(), desc = "found 0", total = name_state.shape[0])
    for idx, row in tqdm_obj:
        airportname, fullstate = row['eventairport_conv'], row['event_fullstate']
        res = search_wiki_airportname(airportname, fullstate)
        if res is not None:
            wiki_search_found[airportname] = res
            tqdm_obj.set_description(f"found {len(wiki_search_found)}")

    wiki_search_found_df = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
    wiki_search_found_df.to_csv('results/wiki_search_found.csv')
else:
    wiki_search_found_df = pd.read_csv('results/wiki_search_found.csv', index_col = 0)

wiki_search_found_set = set(wiki_search_found_df.index)

# all the names we've matched to codes
total_matched_names = all_matched_names.union(wiki_search_found_set)
print(coverage(full.loc[full['eventairport_conv'].apply(lambda x: x in total_matched_names), :].shape[0],
        full.shape[0]))

nf_sel = not_matched['eventairport_conv'].apply(lambda x: x not in total_matched_names)
not_matched.loc[nf_sel, :].to_csv('not_matched.csv', index = False)

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

for idx, row in tqdm(handcoded.iterrows(),total = handcoded.shape[0]):
    if row['tracon_code'] != '?':
        full.loc[(full['eventairport_conv'] == row['eventairport_conv']) & \
                 (full['eventcity'] == row['eventcity']) & \
                 (full['eventstate'] == row['eventstate']), 'tracon_code'] = row['tracon_code']

print('non empty tracon code rows', full.loc[~full['tracon_code'].isna()].shape[0])

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

for idx, row in tqdm(code_and_loc_pd.iterrows(), total = code_and_loc_pd.shape[0]):
    if row['tracon_code'] not in matched_set:
        if check_code(row['tracon_code']):
            matched_set.add(row['tracon_code'])
ct = 0
for code in matched_set:
    sel = tmp['tracon_code'] == code
    ct += tmp.loc[sel].shape[0]
print('wiki + airnav matched', ct)
# ct = 0
# tqdm_obj = tqdm(code_and_loc_pd.iterrows(), desc = 'found 0', total = code_and_loc_pd.shape[0])
# for idx, row in tqdm_obj:
#     tuple_geo = (row['eventcity'], row['event_fullstate'], row['tracon_code'])
#     if tuple_geo not in matched_set:
#         if check_code(row['tracon_code'], row['eventcity'], row['event_fullstate']):
#             ct += 1
#             tqdm_obj.set_description(f"found {ct}")
#             matched_set.add(tuple_geo)
#             if len(matched_set) % 25 == 0:
#                 pickle.dump(matched_set, open('results/faa_matched_set.pckl', 'wb'))

embed()


# hack around groupby ignoring nan values
full['tracon_code'] = full['tracon_code'].fillna('nan') 

vol_tracons = set(pickle.load(open('../results/vol_data.pckl', 'rb')))
num_na = (full['tracon_code'] == 'nan').sum()
print('vol match', coverage(full['tracon_code'].apply(lambda x: x in vol_tracons).sum(), \
        full.shape[0]), f'na codes {num_na}')

cols = ['tracon_code', 'month', 'year', 'eventtype']
full = full[cols].groupby(cols[:-1]).count().rename({cols[-1]: 'faa_incidents'}, axis = 1).reset_index()
full['tracon_code'] = full['tracon_code'].str.replace("none", "")

# deal with na values
full.loc[full['tracon_code'] == 'nan', 'tracon_code'] = np.nan
full.loc[full['tracon_code'] == 'none', 'tracon_code'] = np.nan

full.to_csv('results/FAA_AIDS_full_processed.csv', index = False)
