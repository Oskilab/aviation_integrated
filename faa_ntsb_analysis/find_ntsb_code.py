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
page_google, search_wiki = False, True
match, create_backup = False, True
key = 'AIzaSyCR9FRYW-Y7JJbo4hU682rn6kJaUA5ABUc'
coverage = namedtuple('coverage', ['part', 'total'])

# load us state abbreviations
us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']

us_to_abbrev = {v: k for k,v in us_state_abbrev.items()}

# preprocess data
full = pd.read_csv('datasets/NTSB_AviationData_new.txt', sep = "|")
strip_cols = [' Airport Code ' , ' Airport Name ', ' Latitude ', ' Longitude ']
for col in strip_cols:
    full[col] = full[col].str.strip()

# process airport code
invalid_codes = ['n/a', 'none', 'na']
for inv_code in invalid_codes:
    full[' Airport Code '] = full[' Airport Code '].str.replace(inv_code, "")
nonna_code_sel = ~(full[' Airport Code '].isna())
full.loc[nonna_code_sel, ' Airport Code '] = full.loc[nonna_code_sel, ' Airport Code '].str.upper()

# process airport name
full[' Airport Name '] = full[' Airport Name '].str.replace("N/A", "")

# dates
full['year'] = full[' Event Date '].str.split("/").apply(lambda x: x[2]).apply(int)
full['month'] = full[' Event Date '].str.split("/").apply(lambda x: x[0]).apply(int)
print('full dataset size', full.shape[0])
full = full.loc[(full['year'] >= 1988) & (full['year'] < 2020), :]
full = full.reset_index().drop('index', axis = 1)

# preliminary stats
print('airport code missing', full.loc[full[' Airport Code '] != ''].shape[0])
print('airport name missing', full.loc[full[' Airport Name '] != ''].shape[0])
print('airport code or name missing', full.loc[(full[' Airport Name '] != '') | \
        (full[' Airport Code '] != '')].shape[0])
print('airport name and no code missing', full.loc[(full[' Airport Name '] != '') & \
        (full[' Airport Code '] == '')].shape[0])

reg_dash = '([a-z]{1,})[-/]{1}([a-z]{1,})'
pat = re.compile(reg_dash)
def process_ntsb_name(x):
    x = str(x).lower()
    fin = []
    for elem in x.split(" "):
        if elem != "airport" and elem != "-":
            if elem.startswith("(") or elem.startswith("/"):
                elem = elem[1:]
            if elem.endswith(")") or elem.endswith("/") or elem.endswith(",") or elem.endswith("-"):
                elem = elem[:-1]
            # add to fin
            reg_res = pat.match(elem)
            if reg_res is not None:
                fin.append(reg_res.group(1))
                fin.append(reg_res.group(2))
            else:
                fin.append(elem)
    return " ".join(fin)

# calculate number of rows that are covered by volume dataset
# vol_tracons = set(pickle.load(open('../results/vol_data.pckl', 'rb')))
# num_na = (full[' Airport Code '] == '').sum()
# print('vol match', coverage(full[' Airport Code '].apply(lambda x: x in vol_tracons).sum(), \
#         full.shape[0]), f'na codes {num_na}')


all_full_states = set(us_state_abbrev.values())
# process airport name
full['eventairport_conv'] = full[' Airport Name '].apply(process_ntsb_name)
full['eventcity'] = full[' Location '].apply(get_city)
full['event_fullstate'] = full[' Location '].apply(get_state)
full['event_country'] = full[' Location '].apply(get_country)


full.loc[full[' Airport Code '].isna(), ' Airport Code '] = ''

assert(full[' Airport Code '].isna().sum() == 0 and full[' Airport Name '].isna().sum() == 0)

code_and_name_empty = (full[' Airport Code '] == '') & (full[' Airport Name '] == '')
full.loc[code_and_name_empty, :].to_csv('results/discarded_ntsb.csv')

# full = full.loc[~code_and_name_empty, :].copy()

"""
This following section queries google nearby places utilizing latitude/longitude information provided
by the NTSB dataset. The wikipedia query creates a list of airportnames, which we then use to query
airnav.com to find the identifier.
"""

# empty_code_sel = full[' Airport Code '] == ''
# empty_lat = full[' Latitude '] == ''
# empty_lon = full[' Longitude '] == ''
# only_lat_lon = ~(empty_lat | empty_lon) & empty_code_sel
#
# if page_google:
#     unique_lat_lon = full.loc[only_lat_lon, [" Latitude ", " Longitude "]].copy().drop_duplicates()
#     tqdm_obj =  tqdm(unique_lat_lon.iterrows(), total = unique_lat_lon.shape[0], desc = "found 0")
#     errors = {}
#     lat_lon_records = []
#     for idx, row in tqdm_obj:
#         lat, lon = row
#         lat, lon = float(lat), float(lon)
#         map_api_str = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={key}' + \
#             f'&location={quote(str(lat))},{quote(str(lon))}&rankby=distance&type=airport'
#
#         try:
#             response = requests.get(map_api_str.replace(" ",""))
#             response.raise_for_status()
#
#             # access JSON content
#             json = response.json()
#             if json['status'] != 'OK':
#                 errors[idx] = json['status']
#             else:
#                 all_res = []
#                 for idx, results in enumerate(json['results']):
#                     res_dict = {'lat': lat, 'lon': lon}
#                     res_dict['airport_name'] = results['name']
#                     res_dict['airport_lat'] = results['geometry']['location']['lat']
#                     res_dict['airport_lon'] = results['geometry']['location']['lng']
#                     res_dict['vicinity'] = results['vicinity']
#                     all_res.append(pd.Series(res_dict).add_suffix(f"_{idx}"))
#                 res = pd.concat(all_res, axis = 0)
#                 lat_lon_records.append(res)
#                 tqdm_obj.set_description(f"found {len(lat_lon_records)}")
#
#                 # for good measure
#                 if (len(lat_lon_records) % 50) == 0:
#                     pd.DataFrame.from_records(lat_lon_records).to_csv('results/lat_lon_records.csv')
#                     pickle.dump(errors, open('results/idx_errors.pckl', 'wb'))
#         except HTTPError as http_err:
#             errors[idx] = str(http_err)
#         except Exception as err:
#             errors[idx] = str(err)
#     lat_lon_pd = pd.DataFrame.from_records(lat_lon_records)
#     lat_lon_pd.to_csv('results/lat_lon_records.csv')
#     pickle.dump(errors, open('results/idx_errors.pckl', 'wb'))
# else:
#     lat_lon_pd = pd.read_csv('results/lat_lon_records.csv', index_col = 0)
#

# def conv_to_float(x):
#     if x == '':
#         return np.nan
#     else:
#         return round(float(x.strip()), 5)

# create geo_full
# geo_full = full.loc[only_lat_lon, geo_cols].drop_duplicates().copy()
# geo_full[' Latitude '] = geo_full[' Latitude '].apply(conv_to_float)
# geo_full[' Longitude '] = geo_full[' Longitude '].apply(conv_to_float)
#
# # combine with lat_lon_pd
# lat_lon_pd.rename({'lat_0': ' Latitude ', 'lon_0': ' Longitude '}, axis = 1, inplace = True)
# lat_lon_pd[' Latitude '] = lat_lon_pd[' Latitude '].apply(lambda x: round(x, 5))
# lat_lon_pd[' Longitude '] = lat_lon_pd[' Longitude '].apply(lambda x: round(x, 5))
# new_res = lat_lon_pd.merge(geo_full, on = [' Latitude ', ' Longitude '], how = 'left')
# new_res.to_csv('results/selenium_output.csv') # used in seleneium
#
# lat_lon_pd = lat_lon_pd[[' Latitude ', ' Longitude ', 'airport_name_0']]
# lat_lon_pd['airport_name_0'] = lat_lon_pd['airport_name_0'].apply(process_ntsb_name)
#
# def convert_to_float(x):
#     x = x.strip()
#     if x == '':
#         return np.nan
#     else:
#         return float(x)
#
# # add back to original dataset
# full[' Latitude '] = full[' Latitude '].apply(convert_to_float)
# full[' Longitude '] = full[' Longitude '].apply(convert_to_float)
# full = full.merge(lat_lon_pd, on = [' Latitude ', ' Longitude '], how = 'left')


# ct = 0
# tqdm_obj = tqdm(code_and_loc_pd.iterrows(), desc = 'found 0', total = code_and_loc_pd.shape[0])
# for idx, row in tqdm_obj:
#     tuple_geo = (row['eventcity'], row['event_fullstate'], row['event_country'], row[' Airport Code '])
#     if tuple_geo not in matched_set:
#         if check_code(row[' Airport Code '], row['eventcity'], row['event_fullstate']):
#             ct += 1
#             tqdm_obj.set_description(f"found {ct}")
#             matched_set.add(tuple_geo)
#             if len(matched_set) % 25 == 0:
#                 pickle.dump(matched_set, open('results/matched_set.pckl', 'wb'))
# embed()
# 1/0

# REMOVED
# look at rows with missing latitude/longitude and see if there's only 1 matching airport
# in the wikipedia database in that location
# wiki_tables = load_full_wiki(us_only = False)
# lat_lon_missing = (empty_lat | empty_lon) & empty_code_sel
# city_state_country = full.loc[lat_lon_missing, \
#         ['eventcity', 'event_fullstate', 'event_country']].drop_duplicates()
# idx_to_iata = {}
# for idx, row in tqdm(city_state_country.iterrows(), total = city_state_country.shape[0]):
#     city, state, country = row
#     search_nonus = wiki_tables.loc[(wiki_tables['wiki_city'] == city) & \
#             (wiki_tables['wiki_country'] == country), :]
#     search_us = wiki_tables.loc[(wiki_tables['wiki_city'] == city) & \
#             (wiki_tables['wiki_country'] == country) & \
#             (wiki_tables['wiki_fullstate'] == state), :]
#     if country != 'United States' and search_nonus.shape[0] == 1:
#         idx_to_iata[idx] = search_nonus['wiki_IATA'].iloc[0]
#     elif country == 'United States' and search_us.shape[0] == 1:
#         idx_to_iata[idx] = search_us['wiki_IATA'].iloc[0]
# print('only airport in city', len(idx_to_iata))
# for key in idx_to_iata.keys():
#     full.loc[key, ' Airport Code '] = idx_to_iata[key]

# use selenium code to fill in results of google query
# import pickle
# idx_results = pickle.load(open('results/idx_results.pckl', 'rb'))
#
# num_total_added = 0
# for idx in tqdm(idx_results.keys(), total = len(idx_results)):
#     lat, lon = new_res.loc[idx, [' Latitude ', ' Longitude ']]
#     lat_sel = full[' Latitude '].apply(lambda x: round(x, 5)) == lat
#     lon_sel = full[' Longitude '].apply(lambda x: round(x, 5)) == lon
#     num_total_added += full.loc[lat_sel & lon_sel, ' Airport Code '].shape[0]
#     full.loc[lat_sel & lon_sel, ' Airport Code '] = idx_results[idx]
# print('selenium added', num_total_added)

"""
look for nearest airports using wac
"""
def search_wac_row(full, row):
    lat, lon = row[' Latitude '], row[' Longitude ']

    res = search_city(row['eventcity'], row['event_country'], \
            row[' Latitude '], row[' Longitude '])
    if res is not None:
        lat_sel = full[' Latitude '].apply(lambda x: round(x, 5)) == lat
        lon_sel = full[' Longitude '].apply(lambda x: round(x, 5)) == lon

        tmp = full.loc[lat_sel & lon_sel, :].copy()

        # add new columns
        for col in res.columns:
            tmp[col] = res.loc[idx, col]

        return tmp, tmp.shape[0]
    return None, 0

empty_code_sel = full[' Airport Code '] == ''
empty_lat, empty_lon = full[' Latitude '] == '', full[' Longitude '] == ''
only_lat_lon = ~(empty_lat | empty_lon) & empty_code_sel

if create_backup:
    backup_ntsb, ct = [], 0
    geo_cols = [' Latitude ', ' Longitude ', 'eventcity', 'event_country']

    tmp_obj = full.loc[only_lat_lon, geo_cols].drop_duplicates()
    tqdm_obj = tqdm(tmp_obj.iterrows(), total = tmp_obj.shape[0], \
            desc = "wac near airports found 0")
    for idx, row in tqdm_obj:
        res, found_ct = search_wac_row(full, row)
        ct += found_ct

        if res is not None:
            backup_ntsb.append(res)
            tqdm_obj.set_description(f"wac near airports found {ct}")

    pd.concat(backup_ntsb, axis = 0, ignore_index = True).to_csv('results/backup_ntsb.csv')
    
"""
Search name via wikipedia
"""
name_nocode = (full[' Airport Code '] == '') & (full[' Airport Name '] != '')

if search_wiki:
    wiki_search_found, total_ct = {}, 0
    name_and_state = full.loc[name_nocode, [' Airport Name ', 'event_fullstate', \
            'event_country']].drop_duplicates()
    tqdm_obj = tqdm(name_and_state.iterrows(), total = name_and_state.shape[0], desc = "wiki found 0, tot 0")
    for idx, row in tqdm_obj:
        airportname, fullstate, country = row[' Airport Name '], row['event_fullstate'], row['event_country']
        if not pd.isna(airportname) and country == 'United States':
            res = search_wiki_airportname(airportname, fullstate)
            if res is not None:
                wiki_search_found[airportname] = res
                total_ct += full.loc[full[' Airport Name '] == airportname].shape[0]
                tqdm_obj.set_description(f"wiki found {len(wiki_search_found)}, tot {total_ct}")
                if len(wiki_search_found) % 25 == 0:
                    wiki_search_found_pd = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
                    wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found.csv')

    wiki_search_found_pd = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
    wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found.csv')
else:
    wiki_search_found_pd = pd.read_csv('results/ntsb_wiki_search_found.csv', index_col = 0)

wiki_search_set = set(wiki_search_found_pd.index)
print(full.loc[full[' Airport Name '].apply(lambda x: x in wiki_search_set), :].shape)

res = match_using_name_loc(full[name_nocode], wiki_tables, col = ' Airport Name ')

print(full[full[' Airport Code '] == ''].shape)
for idx, row in tqdm(res.iterrows(), total = res.shape[0]):
    full.loc[name_nocode & (full[' Airport Code '] == row[' Airport Code ']) & \
            (full['eventcity'] == row['eventcity']) &
            (full['event_fullstate'] == row['event_fullstate']), ' Airport Code '] = row['wiki_IATA']
print(full[full[' Airport Code '] == ''].shape)

"""
Matching codes (double checking they are codes with correct location info via wikipedia
or airnav)
"""
def match_via_wikipedia(full, load_saved = True):
    def match_row(code):
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
        if code in matched_set or code in matched_set:
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
    # searching via selenium
    tqdm_obj = tqdm(code_and_loc_pd.iterrows(), total = code_and_loc_pd.shape[0],\
            desc = f'match wac found {len(matched_set)}')
    for idx, row in tqdm_obj:
        code = row[' Airport Code ']
        # manually skipping
        if idx < 26627:
            continue
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

full['found_code'] = 0
if match:
    # see if the codes are matched in wikipedia table
    wiki_tables = load_full_wiki(us_only = False)
    matched_set, not_matched_set = match_via_wikipedia(full, load_saved = True)

    # count the number of codes matched in wikipedia table
    tmp = full.loc[~code_and_name_empty, :].copy()
    ct = 0
    for code in matched_set:
        ct += tmp.loc[tmp[' Airport Code '] == code].shape[0]
    print('wiki matched', ct)

    match_via_wac(full, matched_set, not_matched_set)

    ct = 0
    for code in matched_set:
        sel = tmp[' Airport Code '] == code
        ct += tmp.loc[sel].shape[0]
    print('wiki + airnav matched', ct)
    pickle.dump(matched_set, open('results/ntsb_matched_set.pckl', 'wb'))
else:
    matched_set = pickle.load(open('results/ntsb_matched_set', 'rb'))

full.loc[full[' Airport Code '].apply(lambda x: x in matched_set), 'found_code'] = 1
"""
Save results
"""
full = pd.concat([full, pd.get_dummies(full[' Investigation Type '], prefix = 'ntsb')], axis = 1)
full = full[[' Airport Code ', 'year', 'month', 'ntsb_ Incident ', 'ntsb_ Accident ']]\
        .groupby([' Airport Code ', 'year', 'month']).sum()

full = full.reset_index()
full.rename({' Airport Code ': 'airport_code'}, axis = 1, inplace = True)
full.loc[full['airport_code'] == 'nan', 'airport_code'] = np.nan
full.to_csv('results/NTSB_AIDS_full_processed.csv', index = False)
