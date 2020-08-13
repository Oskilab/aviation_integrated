#!/usr/bin/env python
# coding: utf-8

import pandas as pd, numpy as np
import ftfy, math
from tqdm import tqdm
from IPython import embed


name_dict_trusted = np.load('results/name_dict.npy',allow_pickle='TRUE').item()
code_dict_trusted = np.load('results/code_dict.npy',allow_pickle='TRUE').item()
code_dict_risky = np.load('results/code_dict_risky.npy',allow_pickle='TRUE').item()


try: 
    name_dict_trusted.pop('')
except: 
    pass
try: 
    code_dict_trusted.pop('')
except: 
    pass
try: 
    code_dict_risky.pop('')
except: 
    pass


"""
Phase 1: Complement missing airport name and code using latitude/longitude
"""

data = pd.read_csv("datasets/NTSB_AIDS_full.txt", sep="|")
ntsb_code = pd.read_csv("datasets/NTSB_airportcode.csv")
ntsb_name = pd.read_csv("datasets/NTSB_airportname.csv")
airport_code = pd.read_csv("datasets/airports.csv", usecols=['ident', 'type', 'name', 'latitude_deg', \
        'longitude_deg', 'iso_country', 'iso_region', 'municipality', 'iata_code', 'local_code'])


airport_code['latitude_deg_rounded'] = np.round(airport_code['latitude_deg'], decimals=1)
airport_code['longitude_deg_rounded'] = np.round(airport_code['longitude_deg'], decimals=1)


new_lat, new_lon = [], []
for lat in data[' Latitude ']:
    try:
        new_lat.append(np.round(float(lat.strip()), decimals=1))
    except: 
        new_lat.append(999999)
for lon in data[' Longitude ']: 
    try:
        new_lon.append(np.round(float(lon.strip()), decimals=1))
    except: 
        new_lon.append(999999)
data['latitude_rounded'] = new_lat
data['longitude_rounded'] = new_lon


data_merged = data.merge(airport_code, how='inner', left_on='latitude_rounded', right_on='latitude_deg_rounded')
data_merged = data_merged.loc[(data_merged['longitude_rounded'] == data_merged['longitude_deg_rounded']),:]

print('data_merged/data', len(data_merged), len(data))

"""
Phase 2: Merge clean airportcode and airportname fields back. If either the airport name or code are
invalid, then use latitude/longitude information to determine the airport code in NTSB incident/accident
dataset.
"""

# #### Use data_merged

def isinvalid(entry): 
    if pd.isna(entry):
        return True
    if type(entry) == str: 
        if entry.strip() == "": 
            return True
        if entry.strip() == "N/A":
            return True
    return False

for ind in data.index:
    if isinvalid(data.loc[ind, ' Airport Name ']) or isinvalid(data.loc[ind, ' Airport Code ']):
        try: 
            sel = (data_merged['latitude_deg_rounded'] == data.loc[ind, 'latitude_rounded']
                    and data_merged['longitude_deg_rounded'] == data.loc[ind, 'longitude_rounded'])
            search = data_merged.loc[sel, 'iata_code']
            data.loc[ind, ' Airport Code '] = search.iloc[0, :]['iata_code']
            print(search)
        except: 
            pass

"""
Phase 3: Clean the NTSB dataset's codes/names. (1) We fix the ntsb_code/ntsb_name dataset (unicode),
(2) we add in country codes to the airports dataset, and (3) we clean the ntsb incident/accident dataset
specifically the airportcodes/names.
"""

def clean_paren(col):
    toReturn = []
    for string in col: 
        string = string.upper().strip()
        left = string.find('(')
        right = string.find(')')
        if left != -1 and right != -1: 
            try: 
                new_string = string[:left] + string[right+1:]
            except:
                new_string = string[:left]
            toReturn.append(new_string)
        elif left != -1: 
            new_string = string[:left]
            toReturn.append(new_string)
        else: 
            toReturn.append(string)
    return toReturn

def clean(col): 
    toReturn = []
    for string in col:
        toReturn.append(''.join([c.upper() for c in string if c.isalpha() or c.isnumeric()]))
    return toReturn


def fix_unicode(iterable):
    iterable_org = [str(d) for d in iterable]
    iterable_fix_unicode, iterable_failed = [], []
    for elem in iterable_org:
        try:
            iterable_fix_unicode.append(ftfy.fix_text(elem))
        except:
            iterable_failed.append(str(elem))
    assert(len(iterable_failed) == 0)
    return iterable_fix_unicode

# clean ntsb airport codes/names
code_fix_unicode = fix_unicode(ntsb_code["airportcode"])
name_fix_unicode = fix_unicode(ntsb_name["airportname"])

ntsb_code["airportcode"] = clean(code_fix_unicode)
ntsb_name["airportname"] = [string.strip() for string in name_fix_unicode]

# add country codes
country_code = pd.read_csv('datasets/iso_3166_country_codes.csv', encoding='ISO-8859-1')
airport_code = airport_code.merge(country_code, how='left', left_on='iso_country', right_on='alpha2')

# clean ntsb incident/accident data (codes/names)
data['airportcode'] = clean(data[' Airport Code '])
data['airportname'] = [string.strip() for string in data[' Airport Name ']]
data["airportname_cleanedletter"] = clean(clean_paren(data['airportname']))
data.drop([' Airport Code ', ' Airport Name '], axis=1, inplace=True)


"""
Phase 4: Use geolocation information (from airports/country_code dataset) to fill in 
airport names and airport codes within the ntsb incident/accident dataset. This is the
first pass and uses latitude/longitude within 1 unit.
"""

missing = []
for i in zip([isinvalid(entry) for entry in data['airportname']], \
        [isinvalid(entry) for entry in data['airportcode']]):
    missing.append(i[0] and i[1])

data_missing = data.loc[missing, :]
for ind in tqdm(data.index): 
    if not (isinvalid(data.loc[ind, 'airportname']) and isinvalid(data.loc[ind, 'airportcode'])): 
        continue
        
    lat = data.loc[ind, 'latitude_rounded']
    lon = data.loc[ind, 'longitude_rounded']
    
    searched = airport_code.loc[airport_code['latitude_deg'] <= math.ceil(lat), :] 
    searched = searched.loc[airport_code['latitude_deg'] >= math.floor(lat), :]
    searched = searched.loc[airport_code['longitude_deg'] <= math.ceil(lon), :] 
    searched = searched.loc[airport_code['longitude_deg'] >= math.floor(lon), :]
    
    if len(searched) == 1: 
        data.loc[ind, 'airportcode'] = searched.loc[searched.index[0], 'iata_code']
        data.loc[ind, 'airportname'] = searched.loc[searched.index[0], 'name_x']
    else: 
        for sind in searched.index: 
            try: 
                if searched.loc[sind, 'municipality'].lower() in data.loc[ind, ' Location '].lower(): 
                    data.loc[ind, 'airportcode'] = searched.loc[sind, 'iata_code']
                    data.loc[ind, 'airportname'] = searched.loc[sind, 'name_x']
                    break
            except: 
                pass

"""
Phase 5: Use geolocation information (from airports/country_code dataset) to fill in 
airport names and airport codes within the ntsb incident/accident dataset. This is the
second pass and it uses country/location information as matching criteria and only replaces
if latitude and longitude information is not available.
"""

for ind in tqdm(data.index): 
    if not (isinvalid(data.loc[ind, 'airportname']) and isinvalid(data.loc[ind, 'airportcode'])): 
        continue

    lat = data.loc[ind, 'latitude_rounded']
    lon = data.loc[ind, 'longitude_rounded']
    country = data.loc[ind, ' Country '].strip()
    location = data.loc[ind, ' Location '].lower().strip()
    try: 
        municipality = location.split(',')[0].strip()
        state = location.split(',')[1].strip()
    except: 
        municipality = location
        state = location
    
    if lat == 999999 or lon == 999999: 
        # check country
        searched =  airport_code.loc[airport_code['name_y'] == country, :]
        
        # check municipality
        for sind in searched.index: 
            try: 
                if searched.loc[sind, 'municipality'].lower() in data.loc[ind, ' Location '].lower(): 
                    if searched.loc[sind, 'iso_region'].split('-')[1].lower() == state: 
                        data.loc[ind, 'airportcode'] = searched.loc[sind, 'iata_code']
                        data.loc[ind, 'airportname'] = searched.loc[sind, 'name_x']
                        break
            except: 
                pass


# #### Match name and code

"""
Phase 6: Match name & code using the dictionaries and create new columns (representing the cleaned
versions of airport_name and airport_code)

Output columns: 
1. airportcode_new
2. airportname_new
3. code valid - 'trusted' if from code_dict_trusted; 'risky' if from code_dict_risky; otherwise 'notfound'
4. name valid - 'trusted' if from name_dict_trusted; otherwise 'notfound'
"""


code_cleaned, name_cleaned, code_bool = [], [], []

for i in data.index: 
    code = data.loc[i, 'airportcode']
    name = data.loc[i, 'airportname_cleanedletter']
    
    code_ind = code_dict_trusted.get(code)
    name_ind = name_dict_trusted.get(code)
    if isinstance(name_ind, pd.Series) and len(name_ind) == 1:
        name_ind = name_ind[0]
    if not pd.isna(code_ind):
        code_cleaned.append(airport_code.loc[code_ind, 'iata_code'])
        name_cleaned.append(airport_code.loc[code_ind, 'name_x'])
        code_bool.append('trusted')
    else: 
        code_ind_r = code_dict_risky.get(code)
        if not pd.isna(code_ind_r) and not pd.isna(name_ind):
            updated = False
            for ind in name_ind: 
                if ind == code_ind_r:      
                    code_cleaned.append(airport_code.loc[code_ind_r, 'iata_code'])
                    name_cleaned.append(airport_code.loc[code_ind_r, 'name_x'])
                    code_bool.append('trusted')
                    updated = True
                    break
            if not updated: 
                code_cleaned.append(airport_code.loc[code_ind_r, 'iata_code'])
                name_cleaned.append(airport_code.loc[code_ind_r, 'name_x'])
                code_bool.append('risky')
        elif not pd.isna(code_ind_r): 
            code_cleaned.append(airport_code.loc[code_ind_r, 'iata_code'])
            name_cleaned.append(airport_code.loc[code_ind_r, 'name_x'])
            code_bool.append('risky')
        elif not pd.isna(name_ind): 
            code_cleaned.append(airport_code.loc[name_ind, 'iata_code'])
            name_cleaned.append(airport_code.loc[name_ind, 'name_x'])
            code_bool.append('trusted')
        else: 
            code_cleaned.append(code)
            name_cleaned.append(name)
            code_bool.append('notfound')


data['airportcode_new'] = code_cleaned
data['airportname_new'] = name_cleaned
data['code valid'] = code_bool

"""
Phase 7: Analysis of the new columns
"""

both_na = []
for i in data.index: 
    both_na.append(isinvalid(data.loc[i, 'airportcode']) and isinvalid(data.loc[i, 'airportname']))


print('total number of entries = ' + str(len(data)))
# print('number of entries with both null code & name = ' + str(sum(both_na)))
print('with trusted code & name = ' + str(len(data[data['code valid']=='trusted'])))
print('with risky code & name = ' + str(len(data[data['code valid']=='risky'])))
print('total notfound = ' + str(len(data[data['code valid']=='notfound'])))

#### Number of accidents per airportcode/airportname per month
"""
Phase 8: Number of accidents/incidents per airport_code/airport_name per month
"""

data = pd.get_dummies(data, columns=[' Investigation Type '], drop_first=True)

# parse date
event_date = np.array([data.loc[i, ' Event Date '].strip().split('/') for i in  np.arange(len(data))]).T
month = event_date[0]
day = event_date[1]
year = event_date[2]

data['year'] = year
data['month'] = month
data.to_csv('results/full_data.csv', index = False)

data_filtered = data[data['code valid'] != 'notfound']
data_filtered['airportcode_new'] = data_filtered.loc[:, 'airportcode_new'].copy().apply(str)
data_filtered['airportname_new'] = data_filtered.loc[:, 'airportname_new'].copy().apply(str)

output_df = data_filtered[['airportcode_new', 'airportname_new', 'year', 'month',\
        ' Investigation Type _ Accident ', ' Investigation Type _ Incident ', 'Event Id ']]\
        .groupby(by=['airportcode_new', 'airportname_new', 'year', 'month']).sum()

output_df.to_csv("results/NTSB_AIDS_full_output_new.csv")
embed()
