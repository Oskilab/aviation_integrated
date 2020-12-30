import argparse
import pickle
from itertools import product
from collections import Counter

from tqdm import tqdm

import pandas as pd
import numpy as np

import preprocess_helper

NUM_TIME_PERIODS = (2020 - 1988) * 12

parser = argparse.ArgumentParser(description='Analyze abbreviations.')
parser.add_argument('-t', action='store_true')
args = parser.parse_args()
test = args.t

# tracon_month columns
sel = ['tracon_code', 'year', 'month']

# columns of interest
cols = ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']

# converts full column name to shortened column name
abrev_col_dict = {'narrative': 'narr', 'synopsis': 'syn', \
        'narrative_synopsis_combined': 'narrsyn', 'combined': 'all', \
        'callback': 'call'}

def convert_ctr_to_series(counter, abrev_set=set(), abrev_col='narr'):
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
    ctr, unique_ct = 0, 0
    for word, num in counter.items():
        if word in abrev_set:
            ctr += num
            unique_ct += 1
    return pd.Series({f'{abrev_col}': ctr, f'unq_{abrev_col}': unique_ct})

def get_selector_for_mult_reports(asrs, other_info):
    """
    Calculates a boolean selector that indicates which rows have multiple reports
    @param: asrs (pd.DataFrame) portion of ASRS dataset
    @param: other_info (dict[colname] -> val), contains miscellaneous information that
        characterizes a given tracon_month. See create_index_dicts for more info
    @returns: a boolean selector that calculates which portion of asrs df has
        multiple reports
    """
    mult_rep_cols = ['narrative', 'callback']
    any_col_has_multiple_reports = None
    for mult_col in mult_rep_cols:
        other_info[f'{mult_col}_num_multiple_reports'] = asrs[f'{mult_col}_multiple_reports'].sum()
        if any_col_has_multiple_reports is None:
            any_col_has_multiple_reports = asrs[f'{mult_col}_multiple_reports']
        else:
            any_col_has_multiple_reports = any_col_has_multiple_reports | \
                    asrs[f'{mult_col}_multiple_reports']
    return any_col_has_multiple_reports

def get_total_num_idents(asrs, other_info, unique_idents):
    tot = 0
    for ident_col in unique_idents:
        num_idents = asrs.loc[:, f'{ident_col}'].sum()
        other_info[f'{ident_col}'] = num_idents
        tot += num_idents
    return tot

def split_on_last_space(trcn_mth):
    """
    Splits word on the last occurence of a space. trcn_mth = f'{trcn} {year}/{month}'
    The issue is that the trcn name might have a space in it, so we split on last occurence
    of space
    @param: trcn_mth (str) format '{trcn} {year}/{month}'
    @returns: list of strs ['{trcn}', '{year}/{month}']
    """
    space_idx = trcn_mth.rindex(' ')
    return [trcn_mth[:space_idx], trcn_mth[space_idx:]]

def to_year(trcn_mth):
    """
    Converts a tracon_month string into a float by extracting the year information
    @param: trcn_mth (str) format '{trcn} {year}/{month}'
    @returns: year
    """
    try:
        return int(float(split_on_last_space(trcn_mth)[1].split("/")[0]))
    except ValueError:
        return np.nan

def to_month(trcn_mth):
    """
    Converts a tracon_month string into a float by extracting the month information
    @param: trcn_mth (str) format '{trcn} {year}/{month}'
    @returns: month
    """
    try:
        return int(float(split_on_last_space(trcn_mth)[1].split("/")[1]))
    except ValueError:
        return np.nan

def add_dates(asrs_df):
    """
    Adds date information to df.
    @param: asrs_df (pd.DataFrame)
    @returns: asrs_df (pd.DataFrame) with date information
    """
    asrs_df['year'] = asrs_df.index.map(to_year)
    asrs_df['month'] = asrs_df.index.map(to_month)
    return asrs_df

def load_asrs_ds():
    """
    Loads ASRS dataset
    @returns: all_pds (pd.DataFrame) ASRS dataset
    """
    all_pds = preprocess_helper.load_asrs(load_saved=True)
    # all_pds = preprocess_helper.tracon_analysis(all_pds)
    return all_pds

def get_ident_ct_cols(all_pds):
    """
    Creates a list of columns with '_ident_ct' in them (these need to be analyzed in the pipeline)
    @param: all_pds (pd.DataFrame) ASRS dataset
    """
    unique_idents = []
    for col in all_pds.columns:
        if "_ident_ct" in col:
            unique_idents.append(col)
    return unique_idents

def load_total_cts(col):
    """
    Loads total_cts dataframe for a specific column. Generated in abbrev_word_analysis.py
    @param: col (str) column to be analyzed
    @returns: total_cts (pd.DataFrame) dataframe generated by abbrev_word_analysis.py
    """
    total_cts = pd.read_csv(f'results/total_cts_tagged_{col}.csv')
    return total_cts.drop('Unnamed: 0', axis=1)

def select_subset(all_pds, tracon_month_df, i):
    """
    This finds the portion of the ASRS dataset with a given tracon_code, year, month.
    @param: all_pds (pd.DataFrame) ASRS dataset
    @param: tracon_month_df (pd.DataFrame) of all unique tracon_code, year, month
        combinations in ASRS dataset (see main.py for how it's generated)
    @param: i (int) the index into tracon_month. This indicates what tracon_code,
        year, month we are querying into all_pds.
    @returns: portion of ASRS dataset with given tracon_code/year/month
    """
    selector = all_pds[sel[0]] == tracon_month_df.loc[i, sel[0]]
    for sel_idx in range(1, len(sel)):
        selector = selector & (all_pds[sel[sel_idx]] == tracon_month_df.loc[i, sel[sel_idx]])
    asrs = all_pds.loc[selector, :].copy()
    return asrs

def create_index_dicts(all_pds, tracon_month_df, col):
    """
    Given a column we are analyzing, we create two dictionaries that maps tracon_months
    to any relevant information we need to track (number of reports, observations, etc.)
    We also keep track of all the words that occur within the tracon_month.
    @param: all_pds (pd.DataFrame) ASRS dataset
    @param: tracon_month_df (pd.DataFrame) of all unique tracon_code, year, month
        combinations in ASRS dataset (see main.py for how it's generated)
    @param: col (str) column we are analyzing
    @returns: index_to_counter (dict[tracon_month_str] -> collections.Counter)
        the counter object keeps track of what words occurred within tracon_month and
        how often each word occurred
    @returns index_to_other_info (dict[tracon_month_str] -> dict[col_name] -> val)
        keeps track of any other relevant information regarding tracon month
    """
    abrev_col = abrev_col_dict[col]
    unique_idents = get_ident_ct_cols(all_pds)

    index_to_counter = {} # dictionary from tracon_month -> collections.Counter obj
    index_to_other_info = {}

    for i in tqdm(range(tracon_month_df.shape[0]), desc=col):
        # this creates the string id of the given tracon_month
        index_id = tracon_month_df.loc[i, sel[0]] + \
                f' {tracon_month_df.loc[i, sel[1]]}/{tracon_month_df.loc[i, sel[2]]}'

        # select the rows of the all_pds dataframe with the given tracon_month
        asrs = select_subset(all_pds, tracon_month_df, i)

        asrs[f'{col}_wc'] = asrs.apply(lambda x: \
                preprocess_helper.convert_to_words(x, col).shape[0], axis=1)

        # fill in additional columns
        other_info = {}
        any_col_has_multiple_reports = get_selector_for_mult_reports(asrs, other_info)

        other_info['avg_code_per_obs'] = asrs['num_code_per_obs'].mean()
        other_info['num_total_idents'] = get_total_num_idents(asrs, other_info, unique_idents)
        other_info['num_multiple_reports'] = any_col_has_multiple_reports.sum()
        other_info['num_observations'] = asrs.shape[0]
        other_info['num_callbacks'] = asrs['contains_callback'].sum()
        other_info[f'{abrev_col}_wc'] = asrs[f'{col}_wc'].sum()
        other_info[f'{abrev_col}_avg_wc'] = asrs[f'{col}_wc'].mean()

        # this is redundant (occurs in preprocess_helper.py)
        asrs[col] = asrs[col].str.lower()

        # this creates a collections.Counter object that counts the number of times each word
        # showed up within the given tracon_month, then saved to index_to_counter
        split = asrs.apply(lambda x: preprocess_helper.convert_to_words(x, col), axis=1)

        index_to_counter[index_id] = Counter(np.hstack(split.values).flatten())
        index_to_other_info[index_id] = pd.Series(other_info)
    return index_to_counter, index_to_other_info

def generate_tracon_month_ctr(col, abrev_set, key_ctr, prefix):
    """
    This generates a dataframe that calculates how many times words from abrev_set occur
    for each tracon_month (as well as how many unique words occur).
    @param: col (str) column we are analyzing
    @param: abrev_set (set of str) words we are interested in
    @param: key_ctr (items of index_to_counter dictionary), each element is a tuple of
        key (of tracon month, e.g., SFO 08/2011) to counter (collections.Counter object)
        see create_index_dicts for more info
    @param: prefix (str) prefix we wish to add to all columns in the final dataframe
    @returns: pd.DataFrame of each tracon_month and the corresponding counts
        e.g, SFO 08/2011 -> 5 occurences of words from abrev_set
    """
    abrev_col = abrev_col_dict[col]
    pos_nonword_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, abrev_set, abrev_col) \
            for key, ctr in key_ctr}, orient='index')
    pos_nonword_df = pos_nonword_df.add_prefix(prefix)
    return pos_nonword_df

def generate_ctr_df(total_cts, index_to_counter, index_to_other_info, col):
    """
    This generates one dataframe that calculates how many times words from each
    abbreviation_dictionaries occur for each tracon_month (each tracon_month has
    an associated casa ct, nasa ct, faa ct, etc.)
    @param: total_cts (pd.DataFrame) dataframe of word counts from ASRS dataset
    @param: index_to_counter (dict[tracon_month_str] -> collections.Counter)
        the counter object keeps track of what words occurred within tracon_month and
        how often each word occurred
    @param index_to_other_info (dict[tracon_month_str] -> dict[col_name] -> val)
        keeps track of any other relevant information regarding tracon month
    @param: col (str) column we are analyzing
    @returns: pd.DataFrame that maps each tracon_month to its associated casa ct/faa ct,
        and counts for each abbreviation dictionary
    """
    aviation_dicts = preprocess_helper.load_dictionaries()

    # set of abbreviations to calculate counts
    all_abrevs = set(total_cts.loc[total_cts['abrev'] == 1, 'acronym'])
    pos_nonword_abrevs = set(total_cts.loc[total_cts['tag'] == 'pos_nonword', 'acronym'])

    # the list of set of acronyms for each dictionary. this is used to create counts for each
    # tracon month (aligned with aviation dicts)
    df_sets = [set(x['acronym']).intersection(all_abrevs) for x in aviation_dicts.values()]

    other_info = pd.DataFrame.from_dict(index_to_other_info, orient='index')
    key_ctr = list(index_to_counter.items())

    pos_nonword_df = generate_tracon_month_ctr(col, pos_nonword_abrevs, key_ctr, "pos_nwrd_")
    all_df = generate_tracon_month_ctr(col, all_abrevs, key_ctr, "abrvs_no_ovrcnt_")

    all_dfs = [pos_nonword_df, all_df, other_info]
    # count the number of times words in each dictionary shows up in each tracon_month
    for dict_idx, dict_name in enumerate(aviation_dicts.keys()):
        cts = generate_tracon_month_ctr(col, df_sets[dict_idx], key_ctr, f"{dict_name}_")
        cts.to_csv(f'results/tracon_month_{col}_{dict_name}.csv')
        all_dfs.append(cts)

    all_dfs = pd.concat(all_dfs, axis=1)
    all_dfs.index = all_dfs.index.rename('tracon_month')

    all_dfs = add_dates(all_dfs)
    return all_dfs

def analyze_wc(all_dfs):
    """
    Calculates columns for total word counts for each tracon_month. TODO change name of this
    function (not just word count anymore)
    @param: all_dfs (pd.DataFrame) maps each tracon_month to its associated word counts
    @returns: adjusted all_dfs with added wc columns
        {abrev_col}_wc_all: total word count of a given time period (year/month combination)
        {abrev_col}_wc_out: word count of all tracons in a given time period outside
            the given tracon_code
    """
    # select all columns that can be summed up (and ignore ident_ct or date columns)
    all_cols = [col for col in all_dfs.columns if 'avg' not in col and 'ident_ct' not in col \
            and col not in ['year', 'month']]

    year_month_gb = all_dfs[['year', 'month'] + all_cols] \
            .groupby(['year', 'month']).sum().reset_index()

    for _, row in year_month_gb.iterrows():
        yr_mth_sel = (all_dfs['year'] == row['year']) & (all_dfs['month'] == row['month'])
        for sum_col in all_cols:
            all_dfs.loc[yr_mth_sel, f'{sum_col}_all'] = row[sum_col]

    for sum_col in all_cols:
        all_dfs[f'{sum_col}_out'] = all_dfs[f'{sum_col}_all'] - all_dfs[sum_col]

    return all_dfs

def load_ntsb_faa_codes():
    """
    Loads np.ndarray of unique trcn_codes from inc_acc dataset (from both NTSB/FAA ds)
    @returns: unique_ntsb_faa_codes (np.ndarray of unique trcn codes from inc_acc ds)
    @returns unique_codes (set of str) set version of unique_ntsb_faa_codes
    """
    unique_code_fn = '../results/unique_airport_code_ntsb_faa.pckl'
    unique_ntsb_faa_codes = pickle.load(open(unique_code_fn, 'rb'))
    unique_codes = set(unique_ntsb_faa_codes)
    return unique_ntsb_faa_codes, unique_codes

def filter_top50(all_dfs, unique_ntsb_faa_codes):
    """
    Only selects rows from all_dfs from top50 iata codes, and filters out unique_ntsb_faa_codes
    as well.
    @param: all_dfs (pd.DataFrame) maps each tracon_month to its associated word counts
    @param: unique_ntsb_faa_codes (np.ndarray of unique trcn codes from inc_acc ds)
    @returns: filtered all_pds (pd.DataFrame)
    @returns: filtered unique_ntsb_faa_codes (np.ndarray)
    """
    top_50_iata = \
            set(pd.read_excel('../datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    all_dfs = all_dfs.loc[all_dfs.index.map(lambda x: x.split()[0] in top_50_iata)]

    unique_ntsb_faa_codes = np.apply_along_axis( \
            lambda x: [elem for elem in x if elem in top_50_iata], \
            0, unique_ntsb_faa_codes)
    return all_dfs, unique_ntsb_faa_codes

def generate_tracon_month_set(all_dfs):
    """
    Generates a set of unique tracon/year/month combinations from the ASRS dataset
    @param: all_dfs (pd.DataFrame) maps each tracon_month to its associated word counts
    @returns: set of unique tracon/year/month combinations from all_dfs (ASRS ds)
    """
    all_combs = set()
    for idx, row in all_dfs.iterrows():
        tracon = idx.split()[0]
        year = row['year']
        month = row['month']

        if isinstance(year, pd.Series):
            year = int(year.iloc[0])
        if isinstance(month, pd.Series):
            month = int(month.iloc[0])

        all_combs.add((tracon, year, month))
    return all_combs

def new_tracon_codes(all_dfs, unique_codes):
    """
    Finds the tracon_codes that are in ASRS dataset, but not in the FAA/NTSB inc/acc dataset.
    @param: all_dfs (pd.DataFrame) maps each tracon_month to its associated word counts
    @param unique_codes (set of str) set of all trcn_codes in FAA/NTSB inc/acc ds.
    """
    asrs_added_tracons = []
    for tracon_code in all_dfs.index.map(lambda x: x.split()[0]).unique():
        if tracon_code not in unique_codes:
            asrs_added_tracons.append(tracon_code)
    return asrs_added_tracons

def generate_missing_df(all_dfs, unique_ntsb_faa_codes, all_combs):
    """
    Create a new dataframe consisting of all tracon_months that are not found in the ASRS
    dataset (but the tracon_code is found in FAA/NTSB inc/acc ds or ASRS)
    @param: all_dfs (pd.DataFrame) maps each tracon_month to its associated word counts
    @param: unique_ntsb_faa_codes (np.ndarray of str) list of all trcn_codes found in
        either the ASRS dataset or the FAA/NTSB inc/acc ds.
    @param: all_combs (set of tuple of trcn_code, year, month), set of all combinations
        of trcn_code/year/month that were found in ASRS dataset
    @returns: dataframe (pd.DataFrame) of all tracon_months not found in ASRS dataset,
        but the trcn_code was found in some dataset.
    """
    new_output = {}
    for tracon, month, year in tqdm(\
            product(unique_ntsb_faa_codes, range(1, 13), range(1988, 2020)),\
            desc='adding empty rows', \
            total=NUM_TIME_PERIODS * unique_ntsb_faa_codes.shape[0]):
        if (tracon, year, month) not in all_combs:
            index = f'{tracon} {year}/{month}'
            new_output[index] = pd.Series(index=all_dfs.columns, dtype=float)
    return pd.DataFrame.from_dict(new_output, orient='index')

def add_missing_rows(all_dfs):
    """
    This finds all tracon_months that are not found in the ASRS dataset, but contains
    a tracon_code that was found in FAA/NTSB inc/acc dataset or the ASRS dataset itself.
    Then, empty rows with those tracon_months are added to our final dataset.
    @param: all_dfs (pd.DataFrame) maps each tracon_month to its associated word counts
    @returns: final dataset with all counts.
    """
    unique_ntsb_faa_codes, unique_codes = load_ntsb_faa_codes()
    all_dfs, unique_ntsb_faa_codes = filter_top50(all_dfs, unique_ntsb_faa_codes)

    all_combs = generate_tracon_month_set(all_dfs)
    asrs_added_tracons = new_tracon_codes(all_dfs, unique_codes)

    # final list of trcn_codes
    unique_ntsb_faa_codes = np.hstack([unique_ntsb_faa_codes, np.array(asrs_added_tracons)])

    return all_dfs.append(generate_missing_df(all_dfs, unique_ntsb_faa_codes, all_combs))

def main():
    all_pds = load_asrs_ds()

    # this groups by tracon_code, year and month, so each row is a unique tracon_month
    tracon_month_df = all_pds[sel].groupby(sel).count().reset_index()

    for col in cols:
        total_cts = load_total_cts(col)

        # create dictionaries mapping from tracon_month to cts
        index_to_counter, index_to_other_info = create_index_dicts(all_pds, \
                tracon_month_df, col)

        # generate count dataframes
        all_dfs = generate_ctr_df(total_cts, index_to_counter, index_to_other_info, col)
        all_dfs = analyze_wc(all_dfs)

        # add rows for missing tracon_months
        all_dfs = add_missing_rows(all_dfs)

        # post-process and save
        all_dfs.drop(['year', 'month'], axis=1, inplace=True)
        all_dfs.to_csv(f'results/tracon_month_{col}.csv', index=True)

if __name__ == "__main__":
    main()
