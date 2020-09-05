from collections import namedtuple
from tqdm import tqdm
from IPython import embed
from urllib.parse import quote
from urllib.error import HTTPError
from common_funcs import load_full_wiki, match_using_name_loc, search_wiki_airportname, \
        get_city, get_state, get_country
import requests, pandas as pd, urllib.request as request, pickle
import re, numpy as np, pickle
page_google, search_wiki = False, False
key = 'AIzaSyCR9FRYW-Y7JJbo4hU682rn6kJaUA5ABUc'
coverage = namedtuple('coverage', ['part', 'total'])

# load us state abbreviations
us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']

# preprocess data
full = pd.read_csv('datasets/NTSB_AviationData_new.txt', sep = "|")
strip_cols = [' Airport Code ' , ' Airport Name ', ' Latitude ', ' Longitude ']
for col in strip_cols:
    full[col] = full[col].str.strip()

invalid_codes = ['n/a', 'none', 'na']
for inv_code in invalid_codes:
    full[' Airport Code '] = full[' Airport Code '].str.replace(inv_code, "")

nonna_code_sel = ~(full[' Airport Code '].isna())
full.loc[nonna_code_sel, ' Airport Code '] = full.loc[nonna_code_sel, ' Airport Code '].str.upper()
full[' Airport Name '] = full[' Airport Name '].str.replace("N/A", "")
full['year'] = full[' Event Date '].str.split("/").apply(lambda x: x[2]).apply(int)
full['month'] = full[' Event Date '].str.split("/").apply(lambda x: x[0]).apply(int)
full = full.loc[(full['year'] >= 1988) & (full['year'] < 2020), :]
full = full.reset_index().drop('index', axis = 1)

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
# vol_tracons = set(pickle.load(open('../results/vol_data.pckl', 'rb')))
# num_na = (full[' Airport Code '] == '').sum()
# print('vol match', coverage(full[' Airport Code '].apply(lambda x: x in vol_tracons).sum(), \
#         full.shape[0]), f'na codes {num_na}')
# def get_city(x):
#     x_split = x.split(", ")
#     if len(x_split) != 2:
#         return np.nan
#     else:
#         return x_split[0].lower().strip()
# def get_state(x):
#     x_split = x.split(", ")
#     if len(x_split) != 2:
#         return np.nan
#     else:
#         state_str = x_split[1].strip()
#         return us_state_abbrev.get(state_str, state_str)

all_full_states = set(us_state_abbrev.values())
full['eventairport_conv'] = full[' Airport Name '].apply(process_ntsb_name)
full['eventcity'] = full[' Location '].apply(get_city)
full['event_fullstate'] = full[' Location '].apply(get_state)
full['event_country'] = full[' Location '].apply(get_country)
full.loc[full[' Airport Code '].isna(), ' Airport Code '] = 'nan'


empty_code_sel = full[' Airport Code '] == ''
empty_lat = full[' Latitude '] == ''
empty_lon = full[' Longitude '] == ''
only_lat_lon = ~(empty_lat | empty_lon) & empty_code_sel

if page_google:
    unique_lat_lon = full.loc[only_lat_lon, [" Latitude ", " Longitude "]].copy().drop_duplicates()
    tqdm_obj =  tqdm(unique_lat_lon.iterrows(), total = unique_lat_lon.shape[0], desc = "found 0")
    errors = {}
    lat_lon_records = []
    for idx, row in tqdm_obj:
        lat, lon = row
        lat, lon = float(lat), float(lon)
        map_api_str = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={key}' + \
            f'&location={quote(str(lat))},{quote(str(lon))}&rankby=distance&type=airport'

        try:
            response = requests.get(map_api_str.replace(" ",""))
            response.raise_for_status()

            # access JSON content
            json = response.json()
            if json['status'] != 'OK':
                errors[idx] = json['status']
            else:
                all_res = []
                for idx, results in enumerate(json['results']):
                    res_dict = {'lat': lat, 'lon': lon}
                    res_dict['airport_name'] = results['name']
                    res_dict['airport_lat'] = results['geometry']['location']['lat']
                    res_dict['airport_lon'] = results['geometry']['location']['lng']
                    res_dict['vicinity'] = results['vicinity']
                    all_res.append(pd.Series(res_dict).add_suffix(f"_{idx}"))
                res = pd.concat(all_res, axis = 0)
                lat_lon_records.append(res)
                tqdm_obj.set_description(f"found {len(lat_lon_records)}")

                # for good measure
                if (len(lat_lon_records) % 50) == 0:
                    pd.DataFrame.from_records(lat_lon_records).to_csv('results/lat_lon_records.csv')
                    pickle.dump(errors, open('results/idx_errors.pckl', 'wb'))
        except HTTPError as http_err:
            errors[idx] = str(http_err)
        except Exception as err:
            errors[idx] = str(err)
    lat_lon_pd = pd.DataFrame.from_records(lat_lon_records)
    lat_lon_pd.to_csv('results/lat_lon_records.csv')
    pickle.dump(errors, open('results/idx_errors.pckl', 'wb'))
else:
    lat_lon_pd = pd.read_csv('results/lat_lon_records.csv', index_col = 0)

geo_full = full.loc[only_lat_lon, [' Latitude ', ' Longitude ', 'eventcity', 'event_fullstate', 'event_country']]\
        .drop_duplicates().copy()
def conv_to_float(x):
    if x == '':
        return np.nan
    else:
        return round(float(x.strip()), 5)

geo_full[' Latitude '] = geo_full[' Latitude '].apply(conv_to_float)
geo_full[' Longitude '] = geo_full[' Longitude '].apply(conv_to_float)

lat_lon_pd.rename({'lat_0': ' Latitude ', 'lon_0': ' Longitude '}, axis = 1, inplace = True)
lat_lon_pd[' Latitude '] = lat_lon_pd[' Latitude '].apply(lambda x: round(x, 5))
lat_lon_pd[' Longitude '] = lat_lon_pd[' Longitude '].apply(lambda x: round(x, 5))

new_res = lat_lon_pd.merge(geo_full, on = [' Latitude ', ' Longitude '], how = 'left')
new_res.to_csv('results/selenium_output.csv')


lat_lon_pd = lat_lon_pd[[' Latitude ', ' Longitude ', 'airport_name_0']]
lat_lon_pd['airport_name_0'] = lat_lon_pd['airport_name_0'].apply(process_ntsb_name)
# lat_lon_pd['iata'] = np.nan
# lat_lon_pd['which_airport'] = np.nan

# ct = 0
# tqdm_obj = tqdm(lat_lon_pd.iterrows(), desc = "found 0 ", total = lat_lon_pd.shape[0])
# for idx, row in tqdm_obj:
#     for i in range(5):
#         airportname = process_ntsb_name(row[f'airport_name_{i}'])
#         if not pd.isna(airportname):
#             res = search_wiki_airportname(airportname)
#             if res is not None:
#                 lat_lon_pd.loc[idx, 'iata'] = res['iata']
#                 lat_lon_pd.loc[idx, 'which_airport'] = airportname
#                 tqdm_obj.set_description(f"found {ct}")
#                 ct += 1
#                 if ct % 25 == 0:
#                     lat_lon_pd.to_csv('results/updated_lat_lon_records.csv')
#                 break
# lat_lon_pd.to_csv('results/updated_lat_lon_records.csv')
# embed()
#
def convert_to_float(x):
    x = x.strip()
    if x == '':
        return np.nan
    else:
        return float(x)

full[' Latitude '] = full[' Latitude '].apply(convert_to_float)
full[' Longitude '] = full[' Longitude '].apply(convert_to_float)
full = full.merge(lat_lon_pd, on = [' Latitude ', ' Longitude '], how = 'left')


# look at rows with missing latitude/longitude and see if there's only 1 matching airport
# in the wikipedia database in that location
wiki_tables = load_full_wiki(us_only = False)
lat_lon_missing = (empty_lat | empty_lon) & empty_code_sel
embed()
1/0
city_state_country = full.loc[lat_lon_missing, \
        ['eventcity', 'event_fullstate', 'event_country']].drop_duplicates()
idx_to_iata = {}
for idx, row in tqdm(city_state_country.iterrows(), total = city_state_country.shape[0]):
    city, state, country = row
    search_nonus = wiki_tables.loc[(wiki_tables['wiki_city'] == city) & \
            (wiki_tables['wiki_country'] == country), :]
    search_us = wiki_tables.loc[(wiki_tables['wiki_city'] == city) & \
            (wiki_tables['wiki_country'] == country) & \
            (wiki_tables['wiki_fullstate'] == state), :]
    if country != 'united states' and search_nonus.shape[0] == 1:
        idx_to_iata[idx] = search_nonus['wiki_iata'].iloc[0]
    elif country == 'united states' and search_us.shape[0] == 1:
        idx_to_iata[idx] = search_us['wiki_iata'].iloc[0]

print('only airport in city', len(idx_to_iata))
for key in idx_to_iata.keys():
    full.loc[key, ' Airport Code '] = idx_to_iata[key]

if search_wiki:
    # search the airport name via wikipedia
    wiki_search_found = {}
    total_ct = 0
    name_and_state = full.loc[only_lat_lon, ['airport_name_0', 'event_fullstate']].drop_duplicates()
    tqdm_obj = tqdm(name_and_state.iterrows(), total = name_and_state.shape[0], desc = "found 0, tot 0")
    for idx, row in tqdm_obj:
        airportname, fullstate = row['airport_name_0'], row['event_fullstate']
        # airportname, fullstate = row
        if not pd.isna(airportname):
            res = search_wiki_airportname(airportname, fullstate)
            if res is not None:
                wiki_search_found[airportname] = res
                total_ct += full.loc[full['airport_name_0'] == airportname].shape[0]
                tqdm_obj.set_description(f"found {len(wiki_search_found)} {total_ct}")
                if len(wiki_search_found) % 25 == 0:
                    wiki_search_found_pd = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
                    wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found1.csv')

    wiki_search_found_pd = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
    wiki_search_found_pd.to_csv('results/ntsb_wiki_search_found.csv')
else:
    wiki_search_found_pd = pd.read_csv('results/ntsb_wiki_search_found.csv', index_col = 0)

wiki_search_set = set(wiki_search_found_pd.index)
print(full.loc[full['airport_name_0'].apply(lambda x: x in wiki_search_set), :].shape)

res1 = match_using_name_loc(full[only_lat_lon], wiki_tables, col = 'airport_name_0')
res2 = match_using_name_loc(full[lat_lon_missing], wiki_tables)

print(full[full[' Airport Code '] == ''].shape)
for idx, row in tqdm(res1.iterrows(), total = res1.shape[0]):
    full.loc[only_lat_lon & (full['airport_name_0'] == row['airport_name_0']) & \
            (full['eventcity'] == row['eventcity']) &
            (full['event_fullstate'] == row['event_fullstate']), ' Airport Code '] = row['wiki_IATA']
print(full[full[' Airport Code '] == ''].shape)
for idx, row in tqdm(res2.iterrows(), total = res2.shape[0]):
    full.loc[only_lat_lon & (full['eventairport_conv'] == row['eventairport_conv']) & \
            (full['eventcity'] == row['eventcity']) &
            (full['event_fullstate'] == row['event_fullstate']), ' Airport Code '] = row['wiki_IATA']
print(full[full[' Airport Code '] == ''].shape)
embed()
1/0

full = pd.concat([full, pd.get_dummies(full[' Investigation Type '], prefix = 'ntsb')], axis = 1)
full = full[[' Airport Code ', 'year', 'month', 'ntsb_ Incident ', 'ntsb_ Accident ']]\
        .groupby([' Airport Code ', 'year', 'month']).sum()

full = full.reset_index()
full.rename({' Airport Code ': 'airport_code'}, axis = 1, inplace = True)
full.loc[full['airport_code'] == 'nan', 'airport_code'] = np.nan
full.to_csv('results/NTSB_AIDS_full_processed.csv', index = False)
