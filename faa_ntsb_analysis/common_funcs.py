from bs4 import BeautifulSoup
from collections import namedtuple
from tqdm import tqdm
from urllib.parse import quote_plus, quote
from IPython import embed
import pandas as pd, urllib.request as request
import numpy as np
coverage = namedtuple('coverage', ['part', 'total'])
us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']

def get_city(x):
    x_split = x.split(",")
    if len(x_split) != 2 and len(x_split) != 3:
        return np.nan
    else:
        return x_split[0].lower().strip()

def get_state(x, full = True):
    x_split = x.split(",")
    if len(x_split) == 3:
        second_elem = x_split[1].strip()
        if full:
            return us_state_abbrev.get(second_elem, second_elem)
        return second_elem
    elif len(x_split) == 2:
        return us_state_abbrev.get(x_split[-1].strip(), np.nan)
    else:
        return np.nan # if 2, only city, country

def get_country(x):
    x_split = x.split(",")
    if len(x_split) >= 2:
        last_elem = x_split[-1].strip()
        if last_elem in us_state_abbrev:
            return "United States"
        else:
            if last_elem == "USA":
                return "United States"
            return last_elem
    else:
        return np.nan

def load_full_wiki(us_only = True):
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

    # load wikipedia airport name
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    all_wiki_tables = []
    for char in alphabet:
        wiki_list_link = f'https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_{char}'
        all_wiki_tables.append(load_wiki_table(wiki_list_link))

    # basic preprocessing
    us_wiki_table = pd.concat(all_wiki_tables, axis = 0, ignore_index = True, sort = True)
    us_wiki_table = us_wiki_table.rename({
        us_wiki_table.columns[4]: 'Location served', 
        us_wiki_table.columns[0]: 'Airport name'}, axis = 1)

    if us_only:
        # only select US
        contains_us_sel = us_wiki_table['Location served'].str.contains("United States")
        us_wiki_table = us_wiki_table.loc[contains_us_sel, :].copy()

    # city/state processing
    us_wiki_table['city'] = us_wiki_table['Location served'].apply(get_city)
    us_wiki_table['fullstate'] = us_wiki_table['Location served'].apply(get_state)
    us_wiki_table['country'] = us_wiki_table['Location served'].apply(get_country)

    us_wiki_table.to_csv('results/us_wiki_tables.csv')
    us_wiki_table = us_wiki_table.add_prefix("wiki_")
    return us_wiki_table

def match_using_name_loc(incident_df, wiki_table, col = 'eventairport_conv'):
    needed_cols = [col, 'eventcity', 'event_fullstate']
    unique_loc = incident_df[needed_cols].copy().drop_duplicates()
    full_matched_pd = []
    tqdm_obj = tqdm(unique_loc.iterrows(), total = unique_loc.shape[0], desc = "found 0")
    for idx, row in tqdm_obj:
        if pd.isna(row[col]) or row[col].strip() == '':
            continue
        all_words_in_name = set(str(row[col]).split(" "))
        # calculate the number of words that match with words in the current airport name
        num_words_matched = wiki_table["wiki_Airport name"].str.lower() \
                .apply(lambda row: sum([1 for x in all_words_in_name if x in set(row.split(" "))]))

        # find airports that contain all words in our airport name
        res = wiki_table.loc[num_words_matched == len(all_words_in_name), :]
        if res.shape[0] == 1: # if only one result, select that row
            full_matched_pd.append(pd.concat([unique_loc.loc[idx, :], res.iloc[0, :]], axis = 0))
            tqdm_obj.set_description(f"found {len(full_matched_pd)}")
            tqdm_obj.refresh()
        elif res.shape[0] == 0: # find airports that contain more than one word in common
            more_than_one_word_common = wiki_table.loc[num_words_matched >= 1, :]
            # if only one result, then select that row
            if more_than_one_word_common.shape[0] == 1:
                full_matched_pd.append(
                    pd.concat([unique_loc.loc[idx, :], more_than_one_word_common.iloc[0, :]], axis = 0)
                )
                tqdm_obj.set_description(f"found {len(full_matched_pd)}")
                tqdm_obj.refresh()
            else: # otherwise start over utilizing state and city
                sel = (wiki_table['wiki_fullstate'] == row['event_fullstate']) & \
                        (wiki_table['wiki_city'] == row['eventcity'])
                searched = wiki_table.loc[sel, :]
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
        
    return pd.concat(full_matched_pd, axis = 1).T

def num_words_matched(left_name, right_name):
    left_name = left_name.lower()
    right_name_set = set(right_name.lower().split())
    return sum([1 for word in set(left_name.split()) if word in right_name_set and \
            word != 'airport'])

def extract_iata(wiki_link, fullstate):
    fp = request.urlopen(wiki_link)
    mybytes = fp.read()
    mystr = mybytes.decode("utf8")

    all_state_names = {state_name:mystr.count(state_name) for state_name \
            in us_state_abbrev.values() if state_name in mystr}
    if pd.isna(fullstate):
        fullstate_count = 0
    else:
        fullstate_count = mystr.count(fullstate)
    find_str = 'title="IATA airport code">IATA</a>: <span class="nickname">'
    
    title_idx = mystr.index("<title>")
    title = mystr[title_idx + len("<title>"):]
    title = title[:title.index("<")]

    if find_str in mystr and (fullstate_count > 0 or pd.isna(fullstate)):
        idx_of_iata = mystr.index(find_str)
        sel_str = mystr[idx_of_iata + len(find_str):]
        idx_of_lt = sel_str.index("<")
        iata = sel_str[:idx_of_lt]
        return mystr, title, iata, all_state_names
    return mystr, title, None, all_state_names

def search_wiki_airportname(airportname, fullstate):
    wiki_search = "https://en.wikipedia.org/w/index.php?search="
    mystr, title, iata, all_state_names = extract_iata(wiki_search + quote(airportname + " airport"), \
            fullstate)
    if iata is not None:
        return pd.Series({'iata': iata, 'wiki_title': title, 'state_names': all_state_names, \
                'num_words_matched': num_words_matched(airportname, title), \
                'max_state': fullstate == max(all_state_names, key = lambda key: all_state_names[key],\
                    default = np.nan)})
    else:
        # if the name does not redirect to a wikipedia page, then it will give a page
        # full of results, which we parse here. We only look at the first five results
        bs4 = BeautifulSoup(mystr, 'html.parser')
        all_res = bs4.find_all('div', {'class': 'mw-search-result-heading'})
        for res in all_res[:5]:
            a_tag = res.a
            try:
                href = a_tag['href']
                if 'Airport' in href or 'airport' in href:
                    wiki_str = 'https://en.wikipedia.org' + a_tag['href']
                    mystr, title, iata, all_state_names = extract_iata(wiki_str, fullstate)
                    if iata is not None:
                        return pd.Series({'iata': iata, 'wiki_title': title, 'state_names': \
                                all_state_names, 'num_words_matched': num_words_matched(airportname, title),\
                                'max_state': fullstate == max(all_state_names, \
                                    key = lambda key: all_state_names[key], default = np.nan) \
                                })
            except KeyError:
                pass
