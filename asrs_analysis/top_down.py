import pandas as pd, numpy as np, re, multiprocessing as mp, gc
import argparse
from IPython import embed
from tqdm import tqdm
from collections import Counter
from preprocess_helper import *

## python top_down.py -all -> analyze the whole corpus + create outputs for abbreviation
## analysis

## python top_down.py -memsafe -> only get relevant output
"""
python top_down.py --job dataset --abrev_cts_path ./results/total_cts_tagged_narrative.csv
"""

parser = argparse.ArgumentParser(description = "Tracon_Month Analysis")
parser.add_argument("--job", nargs = 1)
parser.add_argument("--abrev_cts_path", nargs = "?")

args = parser.parse_args()

if (len(args.job) == 0) or args.job[0] != 'abrev_analysis' and args.job[0] != 'dataset':
    print('must include --job argument. -h for more info')
    quit()
else:
    job = args.job[0]

if job == 'dataset':
    abrev_cts_path = args.abrev_cts_path

total_cts = pd.read_csv(abrev_cts_path)
total_cts = total_cts.drop('Unnamed: 0', axis = 1)
all_abrevs = set(total_cts.loc[total_cts['abrev'] == 1, 'acronym'])
pos_nonword_abrevs = set(total_cts.loc[total_cts['tag'] == 'pos_nonword', 'acronym'])

import copy
def convert_ctr_to_series(counter, abrev_set = set()):
    ct, unique_ct = 0, 0
    for word, num in counter.items():
        if word in abrev_set:
            ct += num
            unique_ct += 1
    return pd.Series({'ct': ct, 'unique_ct': unique_ct})

all_pds = load_asrs()
aviation_dicts = load_dictionaries()

level = ['tracon', 'tracon_month', 'tracon_quarter']
all_sel = [['tracon_code'], ['tracon_code', 'year', 'month'], ['tracon_code', 'year', 'quarter']]

df_sets = [set(x['acronym']).intersection(all_abrevs) for x in aviation_dicts.values()]

for idx, sel in list(enumerate(all_sel))[1:2]:
    tmp = all_pds[sel].groupby(sel).count().reset_index()

    for col in ['narrative', 'synopsis', 'combined']:
        print(level[idx], col)
        index_to_counter = {}
        for i in tqdm(range(tmp.shape[0])):
            index_id = tmp.loc[i, sel[0]]
            if len(sel) > 1 and sel[2] == 'month':
                index_id += f' {tmp.loc[i, sel[1]]}/{tmp.loc[i, sel[2]]}'
            elif len(sel) > 1 and sel[2] == 'quarter':
                index_id += f' {tmp.loc[i, sel[1]]}Q{tmp.loc[i, sel[2]]}'
            
            selector = all_pds[sel[0]] == tmp.loc[i, sel[0]]
            for sel_idx in range(1, len(sel)):
                selector = selector & (all_pds[sel[sel_idx]] == tmp.loc[i, sel[sel_idx]])

            asrs = all_pds.loc[selector, :].copy()
            asrs[col] = asrs[col].str.lower()
            split = asrs.apply(lambda x: convert_to_words(x, col), axis = 1)
            index_to_counter[index_id] = Counter(np.hstack(split.values).flatten())

        key_ctr = list(index_to_counter.items())

        pos_nonword_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, pos_nonword_abrevs) \
                for key, ctr in key_ctr}, orient = 'index')
        pos_nonword_df = pos_nonword_df.add_prefix("pos_nonword_")

        all_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, all_abrevs) \
                for key, ctr in key_ctr}, orient = 'index')
        all_df = all_df.add_prefix("all_abrevs_no_overcount_")

        all_dfs = [pos_nonword_df, all_df]
        for dict_idx, dict_name in enumerate(aviation_dicts.keys()):
            print(dict_name)
            cts = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, df_sets[dict_idx])\
                    for key, ctr in key_ctr}, orient = 'index')
            cts = cts.add_prefix(f"{dict_name}_")
            cts.to_csv(f'results/{level[idx]}_{col}_{dict_name}.csv')
            all_dfs.append(cts)
        all_dfs = pd.concat(all_dfs, axis = 1)
        all_dfs.index = all_dfs.index.rename('tracon_month')
        all_dfs.to_csv(f'results/{level[idx]}_{col}.csv', index = True)
