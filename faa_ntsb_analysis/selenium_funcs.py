import pandas as pd, pickle, numpy as np, re, time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located, url_changes
from IPython import embed
import bs4
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from common_funcs import get_city, get_state, get_country, num_words_matched
from math import pi, cos, asin, sqrt
import urllib.request as request
from tqdm import tqdm
from urllib.parse import quote_plus

us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']
us_state_abbrev['U.S.V.I.'] = 'Virgin Islands'
us_to_abbrev = {v: k for k,v in us_state_abbrev.items()}

firefox_settings = Options()
firefox_settings.add_argument('--headless')


num_driver = 0
driver = webdriver.Firefox(options = firefox_settings, log_path = f'./Logs/geckodriver{num_driver}.log')
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


def get_airports_in_city(city, state):
    # driver = webdriver.Firefox(options = firefox_settings)
    should_continue = False
    # while not should_continue:
    #     try:
    driver.get("http://www.airnav.com/airports")
    elem = driver.find_element_by_name("s")
    elem.clear()
    elem.send_keys(f'{city}, {state}, USA')
    elem.send_keys(Keys.RETURN)
    elem_h3 = WebDriverWait(driver, 3).until(
        presence_of_element_located((By.TAG_NAME, "h3"))
    )
    #     should_continue = False
    # except TimeoutException:
    #     driver.quit()
    #     driver = webdriver.Firefox(options = firefox_settings)
    #     time.sleep(0.5)
    # except NoSuchElementException:
    #     driver.quit()
    #     driver = webdriver.Firefox(options = firefox_settings)
    #     time.sleep(0.5)
    # except:
    #     driver.save_screenshot('screenshot.png')
    #     embed()
    #     1/0

    try:
        page_source = str(driver.page_source)
        table = pd.read_html(page_source)[3].iloc[:, [0, 2, 4]]

        new_cols = ['lat', 'lon', 'iata']
        for col in new_cols:
            table[col] = np.nan

        for idx, row in table.iterrows():
            iata, lat, lon = get_info(row['ID'])
            table.loc[idx, 'lat'] = lat
            table.loc[idx, 'lon'] = lon
            table.loc[idx, 'iata'] = iata
        table['eventcity'] = table['City'].apply(get_city)
        table['event_fullstate'] = table['City'].apply(get_state)
        table['event_country'] = table['City'].apply(get_country)
        return table
    except:
        driver.save_screenshot('screenshot.png')
        embed()
        1/0

def get_closest_data(city, state, lat, lon):
    table = get_airports_in_city(city, state)
    if table is None:
        return None

    table['distance_km'] = table.apply(lambda row: distance(lat, lon, row['lat'], row['lon']), axis = 1)
    table.sort_values(by = 'distance_km', ascending = True, inplace = True)
    table = table.reset_index()

    needed_cols = ['iata', 'eventcity', 'event_fullstate', 'event_country', 'distance_km']
    vals = table[needed_cols].values.flatten()
    all_cols = []
    for i in range(table.shape[0]):
        for elem in needed_cols:
            all_cols.append(elem + f"_{i}")

    return pd.Series(vals, index = all_cols) # need to update what happens when this is called

def quit():
    driver.quit()

def check_code(code):
    global driver, num_driver
    # driver = webdriver.Firefox(options = firefox_settings)
    should_continue = False
    ct = 0
    while not should_continue:
        try:
            driver.get("http://www.airnav.com/airports")
            elem = driver.find_element_by_name("s")
            elem.clear()
            elem.send_keys(f'{code}')
            elem.send_keys(Keys.RETURN)
            elem_h3 = WebDriverWait(driver, 3).until(
                presence_of_element_located((By.TAG_NAME, "h3"))
            )
            elem_h3_text = elem_h3.text
            should_continue = True
        except TimeoutException:
            driver.quit()
            num_driver += 1
            driver = webdriver.Firefox(options = firefox_settings, \
                    log_path = f'./Logs/geckodriver{num_driver}.log')
            time.sleep(0.2)
        except NoSuchElementException:
            driver.quit()
            num_driver += 1
            driver = webdriver.Firefox(options = firefox_settings, \
                    log_path = f'./Logs/geckodriver{num_driver}.log')
            time.sleep(0.2)
        except:
            driver.save_screenshot('screenshot.png')
            embed()
            1/0
        ct += 1
        if ct >= 2:
            return False

    if 'airports found' in elem_h3_text:
        return True
    elif 'GOING TO' in elem_h3_text:
        return True
    else:
        return False

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

    table['lat'] = np.nan
    table['lon'] = np.nan
    for idx, row in table.iterrows():
        link = row['links']
        lat, lon = get_loc_wac(link)
        table.loc[idx, 'lat'] = lat
        table.loc[idx, 'lon'] = lon

    try:
        table['distance_km'] = table.apply(lambda row: distance(lat, lon, row['lat'], row['lon']), axis = 1)
    except:
        embed()
        1/0

    table.sort_values(by = 'distance_km', ascending = True, inplace = True)
    table = table.reset_index()

    needed_cols = ['IATA', 'City', 'Country', 'distance_km']
    vals = table[needed_cols].values.flatten()
    all_cols = []
    for i in range(table.shape[0]):
        for elem in needed_cols:
            all_cols.append(elem + f"_{i}")

    return pd.Series(vals, index = all_cols) # need to update what happens when this is called

# print(search_city("san francisco", "United States", 37.5, -122.5))
