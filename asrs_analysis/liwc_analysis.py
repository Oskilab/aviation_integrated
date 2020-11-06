import pandas as pd, numpy as np
import re, copy
from IPython import embed
from tqdm import tqdm
from collections import Counter
from preprocess_helper import *
import argparse, pickle
from itertools import product

num_time_periods = (2020 - 1988) * 12
liwc = pd.read_excel('dictionaries/LIWC2015-dictionary-poster-unlocked.xlsx').iloc[3:]
# asrs = pd.read_csv('datasets/ASRS 1988-2019_extracted.csv')
parser = argparse.ArgumentParser(description='Analyze abbreviations.')
parser.add_argument('-t', action = 'store_true')
args = parser.parse_args()
test = args.t

def convert_ctr_to_series(counter, word_dict = {}):
    # word_dict = copy.copy(word_dict)
    total_ct = 0
    for word, num in counter.items():
        if word in word_dict:
            total_ct += num
            # word_dict[word] = num
    # return pd.Series(word_dict)
    return total_ct

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
def add_dates(df):
    df['year'] = df.index.map(to_year)
    df['month'] = df.index.map(to_month)
    return df
def generate_proportions(fin_df, tqdm_desc = "generate props"):
    fin_df['year'] = fin_df.index.map(to_year)
    fin_df['month'] = fin_df.index.map(to_month)

    fin_df_grouped = fin_df.groupby(['year', 'month']).sum()
    orig_cols = list(fin_df.columns)
    for col1, col2 in fin_df.columns:
        if col1 != 'year' and col1 != 'month':
            fin_df[col1, col2.replace("_ct", "_prop")] = np.nan
    ym_df = fin_df[[('year', ''), ('month', '')]].drop_duplicates()
    for idx, row in tqdm(ym_df.iterrows(), total = ym_df.shape[0], desc = tqdm_desc):
        year, month = row['year', ''], row['month', '']
        if pd.isna(year) or pd.isna(month):
            continue
        sel = (fin_df['year', ''] == year) & (fin_df['month', ''] == month)
        for col1, col2 in orig_cols:
            if col1 != 'year' and col1 != 'month':
                fin_df.loc[sel, (col1, col2.replace("_ct", "_prop"))] = \
                         fin_df.loc[sel, (col1, col2)] / \
                         fin_df_grouped.loc[(year, month), (col1, col2)]
    fin_df.drop([('year', ''), ('month', '')], axis = 1, inplace = True)
    return fin_df

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

def analyze_tracon_period(df_grouped, sel_cols, df, group_to_set, replace_dict, col, abrev_col):
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
                columns = [f"liwc_{liwc_group}_{abrev_col}_ct"])
    fin_df = pd.concat(all_df, axis = 1)
    fin_df = generate_proportions(fin_df, "generate props w/o replace")

    # with replace
    all_df = {}
    for liwc_group in group_to_set.keys():
        all_df[liwc_group]  = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, \
                group_to_set[liwc_group]) for key, ctr in key_ctr_replace}, orient = 'index',\
                columns = [f"liwc_{liwc_group}_flfrm_{abrev_col}_ct"])
    fin_df_replace = pd.concat(all_df, axis = 1)
    fin_df_replace = generate_proportions(fin_df_replace, "generate props w/replace")

    fin = pd.concat([fin_df, fin_df_replace], axis = 1)
    fin = add_dates(fin)

    # add in missing rows
    unique_code_fn = '../results/unique_airport_code_ntsb_faa.pckl'
    unique_ntsb_faa_codes = pickle.load(open(unique_code_fn, 'rb'))
    unique_codes = set(unique_ntsb_faa_codes)

    # if test:
    top_50_iata = set(pd.read_excel('../datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    fin = fin.loc[fin.index.map(lambda x: x.split()[0] in top_50_iata)]
    unique_ntsb_faa_codes = np.apply_along_axis(lambda x: [elem for elem in x if elem in top_50_iata], \
            0, unique_ntsb_faa_codes)

    all_combs = set()
    for idx, row in fin.iterrows():
        tracon = idx.split()[0]
        year = row['year']['']
        month = row['month']['']
        all_combs.add((tracon, year, month))

    asrs_added_tracons = []
    for tracon_code in fin.index.map(lambda x:x.split()[0]).unique():
        if tracon_code not in unique_codes:
            asrs_added_tracons.append(tracon_code)

    unique_ntsb_faa_codes = np.hstack([unique_ntsb_faa_codes, np.array(asrs_added_tracons)])

    new_output = {}
    for tracon, month, year in tqdm(product(unique_ntsb_faa_codes, range(1, 13), range(1988, 2020)), \
            desc = 'adding empty rows', total = num_time_periods * unique_ntsb_faa_codes.shape[0]):
        if (tracon, year, month) not in all_combs:
            index = f'{tracon} {year}/{month}'
            new_output[index] = pd.Series(index = fin.columns)

    fin = fin.append(pd.DataFrame.from_dict(new_output, orient = 'index'))
    fin.drop(['year', 'month'], axis = 1, inplace = True)

    fin.to_csv(f'results/liwc_tracon_month_{col}_counts.csv')

dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'iata']

abrev_col_dict = {'narrative': 'narr', 'synopsis': 'syn', \
        'narrative_synopsis_combined': 'narrsyn', 'combined': 'all', \
        'callback': 'call'}

top_50_iata = set(pd.read_excel('../datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
all_pds = all_pds.loc[all_pds['tracon_code'].apply(lambda x: x in top_50_iata)]
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

    analyze_tracon_period(tmp, sel, all_pds, group_to_set, replace_dict, col, abrev_col_dict[col])
