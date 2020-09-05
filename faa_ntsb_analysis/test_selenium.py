import pandas as pd, pickle
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.expected_conditions import presence_of_element_located, url_changes
from IPython import embed
from bs4 import BeautifulSoup
from selenium.common.exceptions import TimeoutException
from common_funcs import get_city, get_state, get_country, num_words_matched
from tqdm import tqdm
reload_saved = True

# lat_lon_pd = pd.read_csv('results/lat_lon_records.csv', index_col = 0)
lat_lon_pd = pd.read_csv('results/selenium_output.csv', index_col = 0)
lat_lon_pd = lat_lon_pd[~lat_lon_pd['eventcity'].isna()].copy()

firefox_settings = Options()
firefox_settings.add_argument('--headless')

import time
start = time.time()
idx_results = {}
if reload_saved:
    idx_results = pickle.load(open('results/idx_results.pckl', 'rb'))
max_results = max(idx_results, default = 0)

tqdm_obj = tqdm(lat_lon_pd.iterrows(), total = lat_lon_pd.shape[0], desc = "found 0")
driver = webdriver.Firefox(options = firefox_settings)
for idx, row in tqdm_obj:
    if idx < max_results:
        continue
    try:
        if len(idx_results) % 25 == 0:
            pickle.dump(idx_results, open('results/idx_results.pckl', 'wb'))
        should_continue = False
        ct = 0
        while not should_continue:
            try:
                driver.get("http://www.airnav.com/airports")
                elem = driver.find_element_by_name("s")
                should_continue = True
            except:
                print(f'{ct} waiting periods when service unavailable')
                driver.quit()
                driver = webdriver.Firefox(options = firefox_settings)
                should_continue = False
                ct += 1
        elem.clear()
        elem.send_keys(row['airport_name_0'])
        elem.send_keys(Keys.RETURN)
        elem_h3 = WebDriverWait(driver, 3).until(
            presence_of_element_located((By.TAG_NAME, "h3"))
        )
        page_source = str(driver.page_source)
        if "Airport not found" in elem_h3.text:
            pass
        elif "EFFECTIVE" in elem_h3.text or 'GOING TO' in elem_h3.text:
            find_str = 'FAA Identifier'
            if find_str not in page_source:
                continue
            find_idx = page_source.index(find_str) + 30

            source = page_source[find_idx:]
            idx_lt = source.index("<")
            iata = source[:idx_lt]

            idx_results[idx] = iata
            tqdm_obj.set_description(f"found {len(idx_results)}")
        elif "airports found" in elem_h3.text:
            results = pd.read_html(page_source)[3].iloc[:, [0, 2, 4]]
            assert('City' in results and 'ID' in results and 'Name' in results)
            results['eventcity'] = results['City'].apply(get_city)
            results['event_fullstate'] = results['City'].apply(get_state)
            results['event_country'] = results['City'].apply(get_country)
            results = results.loc[results['event_fullstate'] == row['event_fullstate']].copy()
            if results.shape[0] > 0:
                num_match = results['eventcity'].apply(lambda x: num_words_matched(row['eventcity'], x))
                if results.loc[num_match > 0, :].shape[0] == 1:
                    airport_id = results.loc[num_match > 0, 'ID'].iloc[0]
                    idx_results[idx] = airport_id
                    tqdm_obj.set_description(f"found {len(idx_results)}")
                else:
                    # maybe choose closest latitude/longitude
                    continue
            else:
                continue
        else:
            driver.save_screenshot('screenshot.png')
            break
    except TimeoutException:
        continue
    except:
        pickle.dump(idx_results, open('results/idx_results_selenium.pckl', 'wb'))
        driver.save_screenshot('screenshot.png')
    finally:
        time.sleep(2)
print(time.time() - start)
driver.quit()
pickle.dump(idx_results, open('results/idx_results.pckl', 'wb'))
