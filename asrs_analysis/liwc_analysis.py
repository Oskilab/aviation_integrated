import pandas as pd, numpy as np
import re, copy
from IPython import embed
from tqdm import tqdm
from collections import Counter
from preprocess_helper import *

liwc = pd.read_excel('dictionaries/LIWC2015-dictionary-poster-unlocked.xlsx').iloc[3:]
# asrs = pd.read_csv('datasets/ASRS 1988-2019_extracted.csv')

def convert_ctr_to_series(counter, word_dict = {}):
    # word_dict = copy.copy(word_dict)
    total_ct = 0
    for word, num in counter.items():
        if word in word_dict:
            total_ct += num
            # word_dict[word] = num
    # return pd.Series(word_dict)
    return total_ct

# create dictionary of name of liwc group -> set of words
start = 0
group_to_set = {}
for idx in (~liwc.loc[3].isna().to_numpy()).nonzero()[0]:
    name = liwc.iloc[1, idx]
    all_words = []
    for series_idx in range(start, idx):
        series = liwc.iloc[2:, series_idx]
        series = series[~series.isna()]
        all_words.append(series.values.flatten())
    # group_to_set[name] = set(np.hstack(all_words))
    group_to_set[name] = {x: 0 for x in np.hstack(all_words)}
    start = idx

all_pds = load_asrs(load_saved = True)

def analyze_tracon_period(df_grouped, sel_cols, df, group_to_set, replace_dict):
    index_to_counter_replace, index_to_counter = {}, {}
    for i in tqdm(range(df_grouped.shape[0])):
        index_id = df_grouped.loc[i, sel_cols[0]]
        if len(sel_cols) > 1 and sel_cols[2] == 'month':
            index_id += f' {df_grouped.loc[i, sel_cols[1]]}/{df_grouped.loc[i, sel_cols[2]]}'
        elif len(sel_cols) > 1 and sel_cols[2] == 'quarter':
            index_id += f' {df_grouped.loc[i, sel_cols[1]]}Q{df_grouped.loc[i, sel_cols[2]]}'
        
        selector = df[sel_cols[0]] == df_grouped.loc[i, sel_cols[0]]
        for sel_idx in range(1, len(sel_cols)):
            selector = selector & (df[sel_cols[sel_idx]] == df_grouped.loc[i, sel_cols[sel_idx]])

        asrs = df.loc[selector, :].copy()
        asrs[col] = asrs[col].str.lower()

        # replace abrevs
        split = asrs.apply(lambda x: convert_to_words(x, col, replace_dict), axis = 1)
        index_to_counter_replace[index_id] = Counter(np.hstack(split.values).flatten())

        # w/o replace abrevs
        split = asrs.apply(lambda x: convert_to_words(x, col, {}), axis = 1)
        index_to_counter[index_id] = Counter(np.hstack(split.values).flatten())

    key_ctr = list(index_to_counter.items())
    key_ctr_replace = list(index_to_counter_replace.items())

    # w/o replace
    all_df = {}
    for liwc_group in group_to_set.keys():
        all_df[liwc_group]  = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, \
                group_to_set[liwc_group]) for key, ctr in key_ctr}, orient = 'index',\
                columns = [f"LIWC_{liwc_group}_noreplace_abrevs_count"])
    fin_df = pd.concat(all_df, axis = 1)

    # with replace
    all_df = {}
    for liwc_group in group_to_set.keys():
        all_df[liwc_group]  = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, \
                group_to_set[liwc_group]) for key, ctr in key_ctr_replace}, orient = 'index',\
                columns = [f"LIWC_{liwc_group}_replace_abrevs_count"])
    fin_df_replace = pd.concat(all_df, axis = 1)
    fin = pd.concat([fin_df, fin_df_replace], axis = 1)
    fin.to_csv(f'results/liwc_tracon_month_{col}_counts.csv')

dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'iata']

sel = ['tracon_code', 'year', 'month']
tmp = all_pds[sel].groupby(sel).count().reset_index()
for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
    total_cts = pd.read_csv(f'results/total_cts_tagged_{col}.csv', index_col = 0)
    total_cts = total_cts.loc[total_cts['abrev'] == 1, :]

    # generate replace dictionary
    replace_dict = {}
    for idx, row in total_cts.iterrows():
        for dict_name in dictionary_names:
            dict_fullform = str(row[f'{dict_name}_fullform']).lower()
            if not pd.isna(dict_fullform):
                replace_dict[row['acronym']] = dict_fullform
                break

    analyze_tracon_period(tmp, sel, all_pds, group_to_set, replace_dict)
