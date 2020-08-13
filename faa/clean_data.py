import pandas as pd
from fuzzywuzzy import process
import requests
from bs4 import BeautifulSoup as bs
import urllib
import time
headers = {}

# Read input files
full = pd.read_csv('input_csvs/FAA_AIDS_full.csv')
codes = pd.read_csv('input_csvs/FAA_AIDS_airports.csv')

# Generate search queries
eventairports = list(codes['eventairport'])
all_airports = list(full['eventairport'])
all_cities = list(full['eventcity'])
pairs = []
for i in range(len(all_cities)):
	pairs.append((all_cities[i], all_airports[i]))
unique_pairs_raw = list(set(pairs))
unique_pairs = []
for pair in unique_pairs_raw:
    if type(pair[1]) == float:
        pass
    else:
        unique_pairs.append(pair)

# Search for codes and names from IATA.org
guesses = {}
not_found = []
count = 1
for pair in tqdm(unique_pairs):
    if type(pair[0]) == float:
        n = pair[1]
    elif pair[0] in pair[1]:
        n = pair[1]
    else:
        n = '{} {}'.format(pair[0], pair[1])
    url_name = urllib.parse.quote(n)
    request_str = 'https://www.iata.org/en/publications/directories/code-search/?airport.search=' + url_name
    r = requests.get(request_str, headers=headers)
    if r.status_code != 200:
        while True:
            print('retrying...')
            time.sleep(10)
            r = requests.get(request_str, headers=headers)
            if r.status_code == 200:
                    break
    soup = bs(r.text)
    try:
        table = soup.findAll('table', attrs={'class':'datatable'})[1]
        table_body = table.find('tbody')
        rows = table_body.find_all('tr')
        data = []
        for row in rows:
            cols = row.find_all('td')
            cols = [ele.text.strip() for ele in cols]
            data.append([ele for ele in cols if ele])
        options = [i[0] + ' ' + i[2] for i in data]
        index = options.index(process.extractOne(n,options)[0])
        best = data[index]
        code = best[3]
        if best[0] == best[2]:
            clean_name = best[0]
        else:
            clean_name = '{} {}'.format(best[0], best[2])
        guesses[pair] = (code, clean_name)
    except:
            not_found.append(n)
    count += 1

# Add new data to dataframe
guesses_mapping = []
for i in range(len(all_cities)):
    key = (all_cities[i], all_airports[i])
    try:
        guesses_mapping.append(guesses[key])
    except:
        guesses_mapping.append('')
iata_code = []
clean_name = []
for g in guesses_mapping:
    if g == '':
        iata_code.append('')
        clean_name.append('')
    else:
        iata_code.append(g[0])
        clean_name.append(g[1])
full['iata_code'] = iata_code
full['clean_name'] = clean_name

# Setup column names for incidents per month dataframe
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
		 'OCT', 'NOV', 'DEC']
years_raw = [str(i%100) for i in range(1978, 2021)]
years = []
for y in years_raw:
    if len(y) == 1:
        years.append('0' + y)
    else:
        years.append(y)
month_year = []
for y in years:
    for m in months:
        month_year.append(m + '-' + y)

# Count number of incidents
full_iata_codes = set(full['iata_code'])
set_clean_names = []
for c in full_iata_codes:
    set_clean_names.append(clean_name[iata_code.index(c)])
column_entries = [[] for i in range(len(month_year))]
for code in full_iata_codes:
    chunk = full.loc[full['iata_code'] == code]
    localeventdate = list(chunk['localeventdate'])
    localeventdate = [i[3:] for i in localeventdate]
    for lst in column_entries:
        lst.append(0)
    for d in localeventdate:
        column_entries[month_year.index(d)][len(column_entries[month_year.index(d)]) - 1] += 1

# Generate dataframe
faa_incidents_df = pd.DataFrame({'iata_code': list(full_iata_codes), 
    'clean_name': set_clean_names})
for i in range(len(column_entries)):
	faa_incidents_df[month_year[i]] = column_entries[i]
# Output CSV files
faa_incidents_df.to_csv('results/faa_incidents.csv')
full.to_csv('results/FAA_AIDS_full.csv')
