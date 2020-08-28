import pandas as pd, numpy as np, re, ssl
import urllib.request as request
from tqdm import tqdm
from urllib.parse import quote_plus, quote
from collections import namedtuple
from bs4 import BeautifulSoup
from IPython import embed
coverage = namedtuple('coverage', ['part', 'total'])
query_wiki, perform_name_matching = False, True
def load_faa_data():
    full = pd.read_csv('input_csvs/FAA_AIDS_full.csv')
    new = pd.read_csv('input_csvs/FAA_AIDS_addition.csv')

    # rename columns
    rename_dict = {}
    for col in new.columns:
        rename_dict[col] = col.lower().replace(" ", "")

    new.rename(rename_dict, axis = 1, inplace = True)
    return pd.concat([full, new], axis = 0, ignore_index = True, sort = False)

# load datasets
full = load_faa_data()

# load us state abbreviations
us_state_abbrev = pd.read_csv('input_csvs/us_state_abbrev.csv', index_col = 0)
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

def load_wiki_table(wiki_link):
    # load html
    fp = request.urlopen(wiki_link)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    
    # read table
    res = pd.read_html(mystr, header = 0)
    wiki_table = res[0]

    # remove rows that are all the same value (they are used to separate sections of the table)
    wiki_table_remove_rows = wiki_table.eq(wiki_table.iloc[:, 0], axis = 0).all(axis = 1)

    wiki_table = wiki_table.drop(wiki_table_remove_rows.to_numpy().nonzero()[0])
    return wiki_table

if perform_name_matching:
    # load wikipedia airport name
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    all_wiki_tables = []
    for char in tqdm(alphabet):
        wiki_list_link = f'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_{char}'
        all_wiki_tables.append(load_wiki_table(wiki_list_link))

    # basic preprocessing
    all_wiki_tables = pd.concat(all_wiki_tables, axis = 0, ignore_index = True, sort = True)
    all_wiki_tables = all_wiki_tables.rename({
        all_wiki_tables.columns[4]: 'Location served', 
        all_wiki_tables.columns[0]: 'Airport name'}, axis = 1)

    # only select US
    contains_us_sel = all_wiki_tables['Location served'].str.contains("United States")
    us_wiki_tables = all_wiki_tables.loc[contains_us_sel, :].copy()

    # city/state processing
    us_wiki_tables['city'] = \
            us_wiki_tables['Location served'].apply(lambda x: x.split(",")[0].lower().strip())
    us_wiki_tables['fullstate'] = \
            us_wiki_tables['Location served'].apply(lambda x: x.split(",")[1].strip())
    us_wiki_tables.to_csv('results/us_wiki_tables.csv')
    us_wiki_tables = us_wiki_tables.add_prefix("wiki_")

    # I know the next section is a block of garbage, but it works decently well, so I'm scared to change it
    # as it takes quite a long time to run the whole thing 
    unique_loc = full[['eventairport_conv', 'eventcity', 'event_fullstate']].copy().drop_duplicates()
    full_matched_pd = []
    tqdm_obj = tqdm(unique_loc.iterrows(), total = unique_loc.shape[0], desc = "found 0")
    for idx, row in tqdm_obj:
        all_words_in_name = set(row['eventairport_conv'].split(" "))
        # calculate the number of words that match with words in the current airport name
        num_words_matched = us_wiki_tables["wiki_Airport name"].str.lower()\
                .apply(lambda row: sum([1 for x in all_words_in_name if x in set(row.split(" "))]))

        # find airports that contain all words in our airport name
        res = us_wiki_tables.loc[num_words_matched == len(all_words_in_name), :]
        if res.shape[0] == 1: # if only one result, select that row
            full_matched_pd.append(pd.concat([unique_loc.loc[idx, :], res.iloc[0, :]], axis = 0))
            tqdm_obj.set_description(f"found {len(full_matched_pd)}")
            tqdm_obj.refresh()
        elif res.shape[0] == 0: # find airports that contain more than one word in common
            more_than_one_word_common = us_wiki_tables.loc[num_words_matched >= 1, :]
            # if only one result, then select that row
            if more_than_one_word_common.shape[0] == 1:
                full_matched_pd.append(
                    pd.concat([unique_loc.loc[idx, :], more_than_one_word_common.iloc[0, :]], axis = 0)
                )
                tqdm_obj.set_description(f"found {len(full_matched_pd)}")
                tqdm_obj.refresh()
            else: # otherwise start over utilizing state and city
                sel = (us_wiki_tables['wiki_fullstate'] == row['event_fullstate']) & \
                        (us_wiki_tables['wiki_city'] == row['eventcity'])
                searched = us_wiki_tables.loc[sel, :]
                if searched.shape[0] == 1:
                    full_matched_pd.append( \
                        pd.concat([unique_loc.loc[idx, :], searched.iloc[0, :]], axis = 0, sort = False)
                    )
                    tqdm_obj.set_description(f"found {len(full_matched_pd)}")
                    tqdm_obj.refresh()
                else:
                    pass
        else:
            # if multiple airports contain all words in our airport name, utilize state/city info
            search_state = res.loc[res['wiki_fullstate'] == row['event_fullstate'], :]
            if search_state.shape[0] == 1: # only one row w/common state is found
                full_matched_pd.append(
                    pd.concat([unique_loc.loc[idx, :], search_state.iloc[0, :]], axis = 0, sort = False)
                )
                tqdm_obj.set_description(f"found {len(full_matched_pd)}")
                tqdm_obj.refresh()
            elif search_state.shape[0] == 0:
                pass
            else:
                search_city = search_state.loc[search_state['wiki_city'] == row['eventcity'], :]
                if search_city.shape[0] == 1: # only one city was found
                    full_matched_pd.append(
                        pd.concat([unique_loc.loc[idx, :], search_city.iloc[0, :]], axis = 0, sort = False)
                    )
                    tqdm_obj.set_description(f"found {len(full_matched_pd)}")
                    tqdm_obj.refresh()
                elif search_city.shape[0] == 0:
                    pass
                else:
                    pass
        
    full_matched_pd = pd.concat(full_matched_pd, axis = 1).T
    full_matched_pd.to_csv('matched_using_name.csv')
else:
    full_matched_pd = pd.read_csv('matched_using_name.csv', index_col = 0)

all_matched_names = set(full_matched_pd['eventairport_conv'])

# work on those that we could not find codes for
not_matched_sel = full['eventairport_conv'].apply(lambda x: x not in all_matched_names)
not_matched = full.loc[not_matched_sel, ['eventairport_conv', 'eventcity']]\
        .groupby('eventairport_conv').count()\
        .sort_values(by = 'eventcity', ascending = False).reset_index()

def extract_iata(wiki_link):
    fp = request.urlopen(wiki_link)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")
    find_str = 'title="IATA airport code">IATA</a>: <span class="nickname">'
    
    title_idx = mystr.index("<title>")
    title = mystr[title_idx + len("<title>"):]
    title = title[:title.index("<")]

    if find_str in mystr:
        idx_of_iata = mystr.index(find_str)
        sel_str = mystr[idx_of_iata + len(find_str):]
        idx_of_lt = sel_str.index("<")
        iata = sel_str[:idx_of_lt]
        return mystr, title, iata
    return mystr, title, None
if query_wiki:
    wiki_search = "https://en.wikipedia.org/w/index.php?search="
    wiki_search_found = {}
    tqdm_obj = tqdm(not_matched['eventairport_conv'], desc = "found 0")
    for airportname in tqdm_obj:
        mystr, title, iata = extract_iata(wiki_search + quote(airportname + " airport"))
        if iata is not None:
            wiki_search_found[airportname] = pd.Series({'iata': iata, 'wiki_title': title})
            tqdm_obj.set_description(f"found {len(wiki_search_found)}")
        else:
            # if the name does not redirect to a wikipedia page, then it will give a page
            # full of results, which we parse here. We only look at the first result
            bs4 = BeautifulSoup(mystr, 'html.parser')
            all_res = bs4.find_all('div', {'class': 'mw-search-result-heading'})
            if len(all_res) > 0:
                a_tag = all_res[0].a
                try:
                    href = a_tag['href']
                    if 'Airport' in href or 'airport' in href:
                        wiki_str = 'https://en.wikipedia.org' + a_tag['href']
                        mystr, title, iata = extract_iata(wiki_str)
                        if iata is not None:
                            wiki_search_found[airportname] = pd.Series({'iata': iata, 'wiki_title': title})
                            tqdm_obj.set_description(f"found {len(wiki_search_found)}")
                except KeyError:
                    pass

    wiki_search_found_df = pd.DataFrame.from_dict(wiki_search_found, orient = 'index')
    wiki_search_found_df.to_csv('results/wiki_search_found.csv')
else:
    wiki_search_found_df = pd.read_csv('results/wiki_search_found.csv', index_col = 0)

wiki_search_found_set = set(wiki_search_found_df.index)

nf_sel = not_matched['eventairport_conv'].apply(lambda x: x not in wiki_search_found_set)
not_matched.loc[nf_sel, :].to_csv('not_matched.csv', index = False)

# all the names we've matched to codes
total_matched_names = all_matched_names.union(wiki_search_found_set)
print(coverage(full.loc[full['eventairport_conv'].apply(lambda x: x in total_matched_names), :].shape[0],
        full.shape[0]))

name_to_iata = {}
for idx, row in full_matched_pd.iterrows():
    name_to_iata[row['eventairport_conv']] = row['wiki_IATA']
for idx, row in wiki_search_found_df.iterrows():
    name_to_iata[idx] = row['iata']

full['tracon_code'] = np.nan
full['tracon_code'] = full['eventairport_conv'].apply(lambda x: name_to_iata[x] if x in name_to_iata else np.nan)

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


full['month'] = full['localeventdate'].apply(get_month)
full['year'] = full['localeventdate'].apply(get_year)
# hack around groupby ignoring nan values
full['tracon_code'] = full['tracon_code'].fillna('nan') 

cols = ['tracon_code', 'month', 'year', 'eventtype']
full = full[cols].groupby(cols[:-1]).count().rename({cols[-1]: 'faa_incidents'}, axis = 1).reset_index()
full['tracon_code'] = full['tracon_code'].str.replace("none", "")

# deal with na values
full.loc[full['tracon_code'] == 'nan', 'tracon_code'] = np.nan
full.loc[full['tracon_code'] == 'none', 'tracon_code'] = np.nan

full.to_csv('results/FAA_AIDS_full_processed.csv')
