"""
Calculates LIWC counts for tracon_months.
"""
import argparse
from collections import Counter

from tqdm import tqdm

import pandas as pd
import numpy as np

import cos_sim
import preprocess_helper
import top_down

# abrev_col_dict = {'narrative': 'narr', 'synopsis': 'syn', \
#         'narrative_synopsis_combined': 'narrsyn', 'combined': 'all', \
#         'callback': 'call'}

sel = ['tracon_code', 'year', 'month']

# columns of interest
cols = ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']

def convert_ctr_to_series(counter, word_dict={}):
    """
    Takes a collections.Counter object and counts the number of times any word inside
    word_dict occurs (adds all the counts together)
    @param: counter (collections.Counter) that counts the number of times a word occurs
        in tracon_month
    @param: word_dict (dict[word] -> 0), a dictionary of a subset of words
    @returns: total_ct (int) the total count of all words in counter that also are contained
        by word_dict
    """
    total_ct = 0
    for word, num in counter.items():
        if word in word_dict:
            total_ct += num
    return total_ct

def generate_proportions(fin_df, tqdm_desc="generate props"):
    """
    For every combination of tracon_month and LIWC group, calculate the proportion of the LIWC
    group that the tracon_month is responsible for in that year/month combination. E.g.,
    tracon_month = SFO August 2011, LIWC group = Pronouns, calculate the number of pronouns that
    occur in SFO August 2011 and calculate the proportion of that number to the total number
    of pronouns that occur in August 2011.
    @param: fin_df (pd.DataFrame) maps each tracon_month to its associated word counts
    @param: tqdm_desc (str) what string to print out for tqdm progressbar
    @returns: adjusted fin_df with new columns for proportions.
    """
    fin_df['year'] = fin_df.index.map(top_down.to_year)
    fin_df['month'] = fin_df.index.map(top_down.to_month)

    fin_df_grouped = fin_df.groupby(['year', 'month']).sum()
    orig_cols = list(fin_df.columns)
    for col1, col2 in fin_df.columns:
        if col1 not in ['year', 'month']:
            fin_df[col1, col2.replace("_ct", "_all")] = np.nan

    ym_df = fin_df[[('year', ''), ('month', '')]].drop_duplicates()
    for _, row in tqdm(ym_df.iterrows(), total=ym_df.shape[0], desc=tqdm_desc):
        year, month = row['year', ''], row['month', '']
        if pd.isna(year) or pd.isna(month):
            continue
        yrmth_sel = (fin_df['year', ''] == year) & (fin_df['month', ''] == month)

        for col1, col2 in orig_cols:
            if col1 not in ['year', 'month']:
                fin_df.loc[yrmth_sel, (col1, col2.replace("_ct", "_all_ct"))] = \
                         fin_df_grouped.loc[(year, month), (col1, col2)]

    for col1, col2 in orig_cols:
        if col1 not in ['year', 'month']:
            fin_df.loc[:, (col1, col2.replace("_ct", "_out_ct"))] = \
                    fin_df.loc[:, (col1, col2.replace("_ct", "_all_ct"))] -\
                    fin_df.loc[:, (col1, col2)]
    fin_df.drop([('year', ''), ('month', '')], axis=1, inplace=True)
    return fin_df

def generate_ct_df(key_ctr, group_to_set, col, flfrm=False):
    """
    This creates a dataframe that maps tracon_months to LIWC counts. E.g., how many pronouns
    were utilized in SFO August 2011?
    @param: key_ctr (items of index_to_counter dictionary), each element is a tuple of
        key (of tracon month, e.g., SFO 08/2011) to counter (collections.Counter object)
    @param: group_to_set (dict[liwc_group] -> dict[word] -> 0) maps liwc groups to a dictionary
        and that dictionary has all the words in that liwc group. E.g., dict[pronouns] is a
        dictionary of all known english pronouns
    @param: col (str) column we are analyzing
    @param: flfrm (bool) whether or not we replaced abbreviations with their full forms
    @returns: pd.DataFrame that maps each tracon_month to all liwc counts.
    """
    abrev_col = preprocess_helper.ABREV_COL_DICT[col]

    all_df = {}
    for liwc_group in group_to_set:
        if not flfrm:
            columns = [f"lwc_{liwc_group}_{abrev_col}_ct"]
        else:
            columns = [f"lwc_{liwc_group}_ff_{abrev_col}_ct"]

        all_df[liwc_group] = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, \
                group_to_set[liwc_group]) for key, ctr in key_ctr}, orient='index',\
                columns=columns)
    return pd.concat(all_df, axis=1)

def rename_column_dict(fin_df):
    """
    This generateas a dictionary that renames the final output columns in order to shorten
    the length of each column name.
    @param: fin_df (pd.DataFrame) dataframe at the end of the pipeline
    @returns: rename_dict (dict[orig_col] = new_col)
    """
    rename_dict = {}
    replace_dict = {'Focus': 'fcs', 'Affiliation': 'affil', 'Interrog': 'inter'}
    for col1, col2 in fin_df.columns:
        new_col = col2
        for key in replace_dict:
            new_col = new_col.replace(key, replace_dict[key])
        rename_dict[col1, col2] = new_col
    return rename_dict

def analyze_tracon_period(df_grouped, asrs_df, group_to_set, col):
    """
    This analyzes one particular column's liwc usage.
    @param: df_grouped (pd.DataFrame) of ASRS dataset grouped with the columns tracon_code,
        year, month
    @param: asrs_df (pd.DataFrame) the ASRS dataset
    @param: group_to_set (dict[liwc_group] -> dict[word] -> 0) maps liwc groups to a dictionary
        and that dictionary has all the words in that liwc group. E.g., dict[pronouns] is a
        dictionary of all known english pronouns
    @param: col (str) column we are analyzing
    """
    replace_dict = cos_sim.load_replace_dictionary(col)

    index_to_counter_replace, index_to_counter = {}, {}
    for i in tqdm(range(df_grouped.shape[0])):
        index_id = df_grouped.loc[i, sel[0]] + \
                f' {df_grouped.loc[i, sel[1]]}/{df_grouped.loc[i, sel[2]]}'

        asrs = top_down.select_subset(asrs_df, df_grouped, i)
        asrs[col] = asrs[col].str.lower()

        # replace abrevs
        split = asrs.apply(lambda x: preprocess_helper.replace_words(x[col], replace_dict), axis=1)
        index_to_counter_replace[index_id] = Counter(np.hstack(split.values).flatten())

        # w/o replace abrevs
        split = asrs.apply(lambda x: preprocess_helper.replace_words(x[col]), axis=1)
        index_to_counter[index_id] = Counter(np.hstack(split.values).flatten())

    key_ctr = list(index_to_counter.items())
    key_ctr_replace = list(index_to_counter_replace.items())

    # w/o replace
    fin_df = generate_ct_df(key_ctr, group_to_set, col, flfrm=False)
    fin_df = generate_proportions(fin_df, "generate props w/o replace")

    # with replace
    fin_df_replace = generate_ct_df(key_ctr_replace, group_to_set, col, flfrm=True)
    fin_df_replace = generate_proportions(fin_df_replace, "generate props w/replace")

    # postprocess results
    fin = pd.concat([fin_df, fin_df_replace], axis=1)
    fin = top_down.add_dates(fin)
    fin = top_down.add_missing_rows(fin)
    fin.drop(['year', 'month'], axis=1, inplace=True)

    rename_dict = rename_column_dict(fin)
    fin.rename(rename_dict, axis=1, inplace=True)

    fin.to_csv(f'results/liwc_tracon_month_{col}_counts.csv')

def process_liwc_groups():
    """
    Reads the LIWC excel file and creates a dictionary that maps from LIWC group to
    another dictionary containing all the words in that particular LIWC group.
    @returns: group_to_set (dict[liwc_group] -> dict[word] -> 0) maps liwc groups to a dictionary
        and that dictionary has all the words in that liwc group. E.g., dict[pronouns] is a
        dictionary of all known english pronouns
    """
    liwc = pd.read_excel('dictionaries/LIWC2015-dictionary-poster-unlocked.xlsx').iloc[3:]
    # create dictionary of name of liwc group -> set of words
    start = 0
    group_to_set = {}
    # liwc dataframe consists of separate groups. The 3rd row being nan indicates
    # that there's a new group, so we iterate over the groups utilizing the
    # np.nonzero() function
    for idx in (~liwc.loc[3].isna().to_numpy()).nonzero()[0]:
        name = liwc.iloc[1, idx] # the name of the LIWC group

        # generate all the words in the group
        all_words = []
        for series_idx in range(start, idx):
            series = liwc.iloc[2:, series_idx]
            series = series[~series.isna()]
            all_words.append(series.values.flatten())

        # creates a dictionary for the LIWC group, maps from word
        # to counts
        group_to_set[name] = {x: 0 for x in np.hstack(all_words)}
        start = idx
    return group_to_set

def load_asrs_ds():
    """
    This loads the ASRS dataset and filters only the top50 iata codes.
    @returns: all_pds (pd.DataFrame) only the top50 iata code portion of the ASRS dataset.
    """
    all_pds = preprocess_helper.load_asrs(load_saved=True)
    # all_pds = preprocess_helper.tracon_analysis(all_pds)
    top_50_iata = \
            set(pd.read_excel('../datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    all_pds = all_pds.loc[all_pds['tracon_code'].apply(lambda x: x in top_50_iata)]
    return all_pds

def main():
    group_to_set = process_liwc_groups()
    all_pds = load_asrs_ds()

    tracon_month_df = all_pds[sel].groupby(sel).count().reset_index()

    for col in cols:
        analyze_tracon_period(tracon_month_df, all_pds, group_to_set, col)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze abbreviations.')
    parser.add_argument('-t', action='store_true')
    args = parser.parse_args()
    test = args.t

    main()
