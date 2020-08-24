import pandas as pd, numpy as np, re, multiprocessing as mp, gc
import argparse
from IPython import embed
from tqdm import tqdm
from collections import Counter
from preprocess_helper import *


"""
python top_down.py --job dataset --abrev_cts_path ./results/total_cts_tagged_narrative.csv
"""

# # Deal with arguments passed in from commandline
# parser = argparse.ArgumentParser(description = "Tracon_Month Analysis")
# parser.add_argument("--job", nargs = 1)
# parser.add_argument("--abrev_cts_path", nargs = "?")
#
# args = parser.parse_args()
#
# if (len(args.job) == 0) or args.job[0] != 'abrev_analysis' and args.job[0] != 'dataset':
#     print('must include --job argument. -h for more info')
#     quit()
# else:
#     job = args.job[0]
#
# if job == 'dataset':
#     abrev_cts_path = args.abrev_cts_path

import copy
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
def convert_ctr_to_series(counter, abrev_set = set()):
    ct, unique_ct = 0, 0
    for word, num in counter.items():
        if word in abrev_set:
            ct += num
            unique_ct += 1
    return pd.Series({'ct': ct, 'unique_ct': unique_ct})

# see preprocess_helper
all_pds = load_asrs(load_saved = True)
aviation_dicts = load_dictionaries()

level = ['tracon', 'tracon_month', 'tracon_quarter']
all_sel = [['tracon_code'], ['tracon_code', 'year', 'month'], ['tracon_code', 'year', 'quarter']]

mult_rep_cols = ['narrative', 'synopsis', 'callback']
for idx, sel in list(enumerate(all_sel))[1:2]:
    # this groups by tracon_code, year and month, so each row is a unique tracon_month
    tmp = all_pds[sel].groupby(sel).count().reset_index()

    for col in ['narrative', 'synopsis', 'combined']:
        print(level[idx], col)
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
        for i in tqdm(range(tmp.shape[0])):
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

            other_info = {}

            any_col_has_multiple_reports = None
            for col in mult_rep_cols:
                other_info[f'{col}_num_multiple_reports'] = asrs[f'{col}_multiple_reports'].sum()
                if any_col_has_multiple_reports is None:
                    any_col_has_multiple_reports = asrs[f'{col}_multiple_reports']
                else:
                    any_col_has_multiple_reports = any_col_has_multiple_reports | \
                            asrs[f'{col}_multiple_reports']
            other_info['num_multiple_reports'] = any_col_has_multiple_reports.sum()
            other_info['num_observations'] = asrs.shape[0]
            other_info['num_callbacks'] = asrs['contains_callback'].sum()

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
        pos_nonword_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, pos_nonword_abrevs) \
                for key, ctr in key_ctr}, orient = 'index')
        pos_nonword_df = pos_nonword_df.add_prefix("pos_nonword_")

        # count the number of times an abrev shows up in each tracon_month
        all_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, all_abrevs) \
                for key, ctr in key_ctr}, orient = 'index')
        all_df = all_df.add_prefix("all_abrevs_no_overcount_")

        all_dfs = [pos_nonword_df, all_df, other_info]
        # count the number of times words in each dictionary shows up in each tracon_month
        for dict_idx, dict_name in enumerate(aviation_dicts.keys()):
            print(dict_name)
            cts = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, df_sets[dict_idx])\
                    for key, ctr in key_ctr}, orient = 'index')
            cts = cts.add_prefix(f"{dict_name}_")
            cts.to_csv(f'results/{level[idx]}_{col}_{dict_name}.csv')
            all_dfs.append(cts)

        # concatenate the dataframe of each counts together and save
        all_dfs = pd.concat(all_dfs, axis = 1)
        all_dfs.index = all_dfs.index.rename('tracon_month')
        all_dfs.to_csv(f'results/{level[idx]}_{col}.csv', index = True)
