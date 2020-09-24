from IPython import embed
from tqdm import tqdm
from collections import Counter
from preprocess_helper import *

import pandas as pd, numpy as np, re
import argparse, pickle, copy
"""
This converts a collections.Counter object to a pd.Series object. The collections.Counter
counts how many times each word showed up within a tracon_month, and the abrev_set is a
set of words that we wish to count within the tracon_month. The final output includes
a regular count (number of times any word within the set showed up), and a unique_ct (
the number of unique words within the set that showed up).
@param: counter (collections.Counter) that counted the # of times each word showed up
    within a given tracon_month
@param: abrev_set (set) a set of all the words you wish to count
@return: pd.Series with a ct of the number of times any word within the set showed up,
    and a unique_ct (the number of unique words within the set that showed up)
"""
def convert_ctr_to_series(counter, abrev_set = set(), abrev_col = 'narr'):
    ct, unique_ct = 0, 0
    for word, num in counter.items():
        if word in abrev_set:
            ct += num
            unique_ct += 1
    return pd.Series({f'{abrev_col}': ct, f'unq_{abrev_col}': unique_ct})

# see preprocess_helper
all_pds = load_asrs(load_saved = True)
aviation_dicts = load_dictionaries()

unique_idents = []
for col in all_pds.columns:
    if "_ident_ct" in col:
        unique_idents.append(col)

mult_rep_cols = ['narrative', 'callback']

# this groups by tracon_code, year and month, so each row is a unique tracon_month
sel = ['tracon_code', 'year', 'month']
tmp = all_pds[sel].groupby(sel).count().reset_index()

abrev_col_dict = {'narrative': 'narr', 'synopsis': 'syn', \
        'narrative_synopsis_combined': 'narrsyn', 'combined': 'all', \
        'callback': 'call'}

for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
    abrev_col = abrev_col_dict[col]
    index_to_counter = {} # dictionary from tracon_month -> collections.Counter obj
    index_to_other_info = {}

    # load total_cts
    total_cts = pd.read_csv(f'results/total_cts_tagged_{col}.csv')
    total_cts = total_cts.drop('Unnamed: 0', axis = 1)

    # set of abbreviations to calculate counts
    all_abrevs = set(total_cts.loc[total_cts['abrev'] == 1, 'acronym'])
    pos_nonword_abrevs = set(total_cts.loc[total_cts['tag'] == 'pos_nonword', 'acronym'])

    # the list of set of acronyms for each dictionary. this is used to create counts for each
    # tracon month
    df_sets = [set(x['acronym']).intersection(all_abrevs) for x in aviation_dicts.values()]
    for i in tqdm(range(tmp.shape[0]), desc = col):
        # this creates the string id of the given tracon_month
        index_id = tmp.loc[i, sel[0]]
        if len(sel) > 1 and sel[2] == 'month':
            index_id += f' {tmp.loc[i, sel[1]]}/{tmp.loc[i, sel[2]]}'
        elif len(sel) > 1 and sel[2] == 'quarter':
            index_id += f' {tmp.loc[i, sel[1]]}Q{tmp.loc[i, sel[2]]}'
        
        # select the rows of the all_pds dataframe with the given tracon_month
        selector = all_pds[sel[0]] == tmp.loc[i, sel[0]]
        for sel_idx in range(1, len(sel)):
            selector = selector & (all_pds[sel[sel_idx]] == tmp.loc[i, sel[sel_idx]])
        asrs = all_pds.loc[selector, :].copy()

        asrs[f'{col}_wc'] = asrs.apply(lambda x: convert_to_words(x, col).shape[0], axis = 1)

        other_info = {}

        any_col_has_multiple_reports = None
        for mult_col in mult_rep_cols:
            other_info[f'{mult_col}_num_multiple_reports'] = asrs[f'{mult_col}_multiple_reports'].sum()
            if any_col_has_multiple_reports is None:
                any_col_has_multiple_reports = asrs[f'{mult_col}_multiple_reports']
            else:
                any_col_has_multiple_reports = any_col_has_multiple_reports | \
                        asrs[f'{mult_col}_multiple_reports']
        tot = 0 
        for ident_col in unique_idents:
            num_idents = asrs.loc[:, f'{ident_col}'].sum()
            other_info[f'{ident_col}'] = num_idents
            tot += num_idents
        other_info['avg_code_per_obs'] = asrs['num_code_per_obs'].mean()
        
        other_info['num_total_idents'] = tot
        other_info['num_multiple_reports'] = any_col_has_multiple_reports.sum()
        other_info['num_observations'] = asrs.shape[0]
        other_info['num_callbacks'] = asrs['contains_callback'].sum()
        other_info[f'{abrev_col}_wc'] = asrs[f'{col}_wc'].sum()
        other_info[f'{abrev_col}_avg_wc'] = asrs[f'{col}_wc'].mean()

        # this is redundant (occurs in preprocess_helper.py)
        asrs[col] = asrs[col].str.lower()

        # this creates a collections.Counter object that counts the number of times each word
        # showed up within the given tracon_month, then saved to index_to_counter
        split = asrs.apply(lambda x: convert_to_words(x, col), axis = 1) # see preprocess.helper
        index_to_counter[index_id] = Counter(np.hstack(split.values).flatten())
        index_to_other_info[index_id] = pd.Series(other_info)

    other_info = pd.DataFrame.from_dict(index_to_other_info, orient = 'index')
    key_ctr = list(index_to_counter.items())

    # count the number of times pos_nonword shows up in each tracon_month
    pos_nonword_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, pos_nonword_abrevs, abrev_col) \
            for key, ctr in key_ctr}, orient = 'index')
    pos_nonword_df = pos_nonword_df.add_prefix(f"pos_nwrd_")

    # count the number of times an abrev shows up in each tracon_month
    all_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, all_abrevs, abrev_col) \
            for key, ctr in key_ctr}, orient = 'index')
    all_df = all_df.add_prefix("abrvs_no_ovrcnt_")

    all_dfs = [pos_nonword_df, all_df, other_info]
    # count the number of times words in each dictionary shows up in each tracon_month
    for dict_idx, dict_name in enumerate(aviation_dicts.keys()):
        cts = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, df_sets[dict_idx], abrev_col)\
                for key, ctr in key_ctr}, orient = 'index')
        cts = cts.add_prefix(f"{dict_name}_")
        cts.to_csv(f'results/tracon_month_{col}_{dict_name}.csv')
        all_dfs.append(cts)
    # TODO: update dictionary code to have proportions

    # concatenate the dataframe of each counts together and save
    all_dfs = pd.concat(all_dfs, axis = 1)
    all_dfs.index = all_dfs.index.rename('tracon_month')

    def to_year(x):
        try:
            return int(float(x.split()[1].split("/")[0]))
        except:
            return np.nan
    def to_month(x):
        try:
            return int(float(x.split()[1].split("/")[1]))
        except:
            return np.nan


    all_dfs['year'] = all_dfs.index.map(to_year)
    all_dfs['month'] = all_dfs.index.map(to_month)

    year_month_gb = all_dfs[['year', 'month', f'{abrev_col}_wc']].groupby(['year', 'month']).sum().reset_index()
    all_dfs[f'{abrev_col}_wc_all'], all_dfs[f'{abrev_col}_wc_out'] = np.nan, np.nan

    for idx, row in year_month_gb.iterrows():
        all_dfs.loc[(all_dfs['year'] == row['year']) & (all_dfs['month'] == row['month']), f'{abrev_col}_wc_all'] \
                = row[f'{abrev_col}_wc']
    all_dfs[f'{abrev_col}_wc_out'] = all_dfs[f'{abrev_col}_wc_all'] - all_dfs[f'{abrev_col}_wc']
    all_dfs[f'{abrev_col}_wc_prop'] = all_dfs[f'{abrev_col}_wc'] / all_dfs[f'{abrev_col}_wc_all']
    all_dfs.drop(['year', 'month'], axis = 1, inplace = True)

    all_dfs.to_csv(f'results/tracon_month_{col}.csv', index = True)
