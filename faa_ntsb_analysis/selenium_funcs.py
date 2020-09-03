import pandas as pd, pickle, numpy as np, re, time
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located, url_changes
from IPython import embed
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from common_funcs import get_city, get_state, get_country, num_words_matched
from math import pi, cos, asin, sqrt
import urllib.request as request
from tqdm import tqdm

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
    driver = webdriver.Firefox(options = firefox_settings)
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
                    log_path = './Logs/geckodriver{num_driver}.log')
            time.sleep(0.5)
        except NoSuchElementException:
            driver.quit()
            num_driver += 1
            driver = webdriver.Firefox(options = firefox_settings, \
                    log_path = './Logs/geckodriver{num_driver}.log')
            time.sleep(0.5)
        except:
            driver.save_screenshot('screenshot.png')
            embed()
            1/0
        ct += 1
        if ct >= 3:
            return False

    if 'airports found' in elem_h3_text:
        return True
    elif 'GOING TO' in elem_h3_text:
        return True
    else:
        return False

# def check_code(code, city, fullstate):
#     global driver
#     should_continue = False
#     ct = 0
#     while not should_continue:
#         try:
#             driver.get("http://www.airnav.com/airports")
#             elem = driver.find_element_by_name("s")
#             elem.clear()
#             elem.send_keys(f'{code}')
#             elem.send_keys(Keys.RETURN)
#             elem_h3 = WebDriverWait(driver, 3).until(
#                 presence_of_element_located((By.TAG_NAME, "h3"))
#             )
#             if 'airports found' in elem_h3.text:
#                 return False
#             should_continue = True
#         except TimeoutException:
#             driver.quit()
#             driver = webdriver.Firefox(options = firefox_settings)
#             time.sleep(0.5)
#         except NoSuchElementException:
#             driver.quit()
#             driver = webdriver.Firefox(options = firefox_settings)
#             time.sleep(0.5)
#         except:
#             driver.save_screenshot('screenshot.png')
#             embed()
#             1/0
#         ct += 1
#         if ct >= 5 and not should_continue:
#             # print(ct)
#             # print(driver.quit())
#             # print('here')
#             # driver.save_screenshot('screenshot.png')
#             # driver.quit()
#             # driver = webdriver.Firefox(options = firefox_settings)
#             return False
#
#     page_source = str(driver.page_source)
#     while "FAA" not in page_source:
#         page_source = str(driver.page_source)
#         time.sleep(0.5)
#         if 'airports found' in elem_h3.text:
#             return False
#     try:
#         title_idx = page_source.index('<meta name="description"')
#         location_str = page_source[title_idx:]
#         location_str_tmp = location_str[location_str.index("(") + 1: location_str.index(")")]
#
#         # move onto next () pair
#         num_comma = location_str_tmp.count(",")
#         if num_comma == 0:
#             location_str = location_str[location_str.index(")"):]
#             location_str = location_str[location_str.index("(") + 1: location_str.index(")")]
#             if location_str.count(",") == 0:
#                 return False
#         elif num_comma == 1:
#             location_str = location_str_tmp
#             airnav_city, airnav_state = location_str.split(", ")
#         elif num_comma == 2:
#             location_str = location_str_tmp
#             airnav_city, airnav_state, country = location_str.split(", ")
#         else:
#             return False
#
#         airnav_fullstate = us_state_abbrev.get(airnav_state, airnav_state)
#         airnav_city = airnav_city.lower()
#
#         return str(city) in airnav_city and airnav_fullstate == fullstate
#     except:
#         driver.save_screenshot('screenshot.png')
#         embed()
#         1/0
#
# # get_closest_data('san francisco', 'ca', 37.5, -122.5)
# # print(check_code('sfo', 'san francisco', 'California'))
# # def get_closest_id(city, state, lat, lon):
