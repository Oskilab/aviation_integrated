import pandas as pd, pickle, numpy as np, re, time
from IPython import embed
import bs4
from bs4 import BeautifulSoup
from common_funcs import get_city, get_state, get_country, num_words_matched
from math import pi, cos, asin, sqrt
import urllib.request as request
from tqdm import tqdm
from urllib.parse import quote_plus

us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']
us_state_abbrev['U.S.V.I.'] = 'Virgin Islands'
us_to_abbrev = {v: k for k,v in us_state_abbrev.items()}

def distance(lat1, lon1, lat2, lon2):
    p = pi/180
    a = 0.5 - cos((lat2-lat1)*p)/2 + cos(lat1*p) * cos(lat2*p) * (1-cos((lon2-lon1)*p))/2
    return 12742 * asin(sqrt(a)) # km

def get_info(airport_id):
    fp = request.urlopen(f"http://www.airnav.com/airport/{airport_id}")
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")

    identifier_idx = mystr.index("FAA Identifier")
    after_identifier = mystr[identifier_idx:]

    td_idx = after_identifier.index("<TD>")
    after_td = after_identifier[td_idx + 4:]
    iata = after_td[:after_td.index("<")]

    lat_long_idx = after_identifier.index("Lat/Long")
    after_lat_lon = after_identifier[lat_long_idx:]
    lat, lon = after_lat_lon.split("<BR>")[2].split(",")
    lat, lon = float(lat), float(lon)

    return iata, lat, lon

def check_code(code):
    wac_link = f"https://www.world-airport-codes.com/search/?s={quote_plus(code)}"
    fp = request.urlopen(wac_link)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")

    html_str_find = "IATA Code</strong>"
    if html_str_find not in mystr:
        return False
    else:
        idx = mystr.index(html_str_find)
        code_str = mystr[idx + len(html_str_find):]
        span_begin_idx = code_str.index("<span>")
        span_end_idx = code_str.index("</span>")
        found_code = code_str[span_begin_idx + len("<span>"):span_end_idx]
        return found_code == code

"""
Querying world-airport-codes.com
"""

def get_table_wac(wac_link):
    fp = request.urlopen(wac_link)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    try:
        table = pd.read_html(mystr)[0]
    except:
        return mystr, None

    bs = BeautifulSoup(mystr, 'html.parser')
    all_hrefs = []

    tbody_it = bs.find_all("tbody")
    if len(tbody_it) == 0:
        return mystr, None

    try:
        for elem in tbody_it[0]:
            if isinstance(elem, bs4.element.Tag):
                all_hrefs.append(elem.contents[1].contents[1]['href'])
        assert(len(all_hrefs) == table.shape[0])
        table['links'] = all_hrefs
        fp.close()
        return mystr, table
    except:
        return mystr, None

def get_loc_wac(wac_link):
    fp = request.urlopen(wac_link)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")

    search_str = 'data-location="'
    search_idx = mystr.index(search_str)
    loc_str = mystr[search_idx + len(search_str):]
    loc_str = loc_str[:loc_str.index('"')]

    lat, lon = loc_str.split(",")
    lat, lon = float(lat), float(lon)

    fp.close()
    return lat, lon

def clean_city_from_wac(city_str, name = "City"):
    if pd.isna(city_str):
        return city_str
    return city_str.replace(f"{name}: ", "")

def search_city(city, country, lat, lon):
    if pd.isna(city):
        return None
    wac_link = f"https://www.world-airport-codes.com/search/?s={quote_plus(city)}"
    mystr, table = get_table_wac(wac_link)

    if table is None:
        return None

    pat = re.compile('/search/\?s=.*&page=\d')
    res = re.search(pat, mystr)
    if res is not None:
        res = res.group(0)

        idx_eq = res.rfind("=")
        max_num = int(res[idx_eq + 1:])
    else:
        return None

    other_tables = [table]
    for other_page_idx in range(2, max_num + 1):
        mystr, o_table = get_table_wac(f"{wac_link}&page={other_page_idx}")
        other_tables.append(o_table)

    table = pd.concat(other_tables, axis = 0, ignore_index = True)

    # cleaning
    table['City'] = table['City'].apply(clean_city_from_wac)
    table['City'] = table['City'].str.lower()
    table['IATA'] = table['IATA'].apply(lambda x: clean_city_from_wac(x, 'IATA'))
    table['Country'] = table['Country'].apply(lambda x: clean_city_from_wac(x, 'Country'))

    table = table.loc[table['City'].str.contains(city).fillna(False)].copy()
    table = table.loc[table['Country'] == country].copy()
    if table.shape[0] == 0:
        return None

    table['lat'] = np.nan
    table['lon'] = np.nan
    for idx, row in table.iterrows():
        link = row['links']
        lat, lon = get_loc_wac(link)
        table.loc[idx, 'lat'] = lat
        table.loc[idx, 'lon'] = lon

    table['distance_km'] = table.apply(lambda row: distance(lat, lon, row['lat'], row['lon']), axis = 1)

    table.sort_values(by = 'distance_km', ascending = True, inplace = True)
    table = table.reset_index()

    needed_cols = ['IATA', 'City', 'Country', 'distance_km']
    vals = table[needed_cols].values.flatten()
    all_cols = []
    for i in range(table.shape[0]):
        for elem in needed_cols:
            all_cols.append(elem + f"_{i}")

    return pd.Series(vals, index = all_cols) # need to update what happens when this is called
