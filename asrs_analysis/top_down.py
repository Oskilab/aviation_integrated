import argparse
import pickle
import time
from itertools import product
from functools import reduce
from collections import Counter

from IPython import embed
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

# month ranges
MONTH_RGS = [1, 3, 6, 12]

def get_year(trcn_mnth):
    """
    Extracts year from a given string (trcn_mnth)
    @param: trcn_mnth (string) with format 'tracon_code year/month'
    @returns: year (float), nan if format incorrect
    """
    try:
        return int(float(str(trcn_mnth).split("/")[0]))
    except ValueError:
        return np.nan
    except IndexError:
        return np.nan

def get_month(trcn_mnth):
    """
    Extracts month from a given string (trcn_mnth)
    @param: trcn_mnth (string) with format 'tracon_code year/month'
    @returns: month (float), nan if format incorrect
    """
    try:
        return int(float(str(trcn_mnth).split("/")[1]))
    except ValueError:
        return np.nan
    except IndexError:
        return np.nan

def convert_ctr_to_series(counter, abrev_set=None, abrev_col='narr', unq_only=False):
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
    if abrev_set is None:
        abrev_set = set()
    ctr, unique_ct = 0, 0
    for word, num in counter.items():
        if word in abrev_set:
            ctr += num
            unique_ct += 1
    if not unq_only:
        return pd.Series({f'{abrev_col}': ctr})
    else:
        return pd.Series({f'unq_{abrev_col}': unique_ct})

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

def create_index_dicts(all_pds, col, month_range=1, lag=0, unq_only=False):
    """
    Given a column we are analyzing, we create two dictionaries that maps tracon_months
    to any relevant information we need to track (number of reports, observations, etc.)
    We also keep track of all the words that occur within the tracon_month.
    @param: all_pds (pd.DataFrame) ASRS dataset
    @param: col (str) column we are analyzing
    @returns: index_to_counter (dict[tracon_month_str] -> collections.Counter)
        the counter object keeps track of what words occurred within tracon_month and
        how often each word occurred
    @returns index_to_other_info (dict[tracon_month_str] -> dict[col_name] -> val)
        keeps track of any other relevant information regarding tracon month
    """
    # converts full column name to shortened column name
    abrev_col = preprocess_helper.ABREV_COL_DICT[col]
    unique_idents = get_ident_ct_cols(all_pds)

    index_to_counter = {} # dictionary from tracon_month -> collections.Counter obj
    index_to_other_info = {}

    all_pds.sort_values(['year', 'month', 'tracon_code'], inplace=True)

    yr_mth = all_pds[['year', 'month']].copy()
    yr_mth, yr_mth_idx, yr_mth_ct = np.unique(yr_mth.values.astype(int), \
            axis=0, return_index=True, return_counts=True)

    for yr_mth_elem_idx, (year, mth) in tqdm(enumerate(yr_mth), desc=col, total=len(yr_mth)):
        tmp = preprocess_helper.year_month_indices(yr_mth, yr_mth_idx, yr_mth_ct, \
                year, mth, month_range, lag)
        asrs_yr_mth = all_pds.iloc[tmp]

        trcn_codes, trcn_codes_idx, trcn_codes_ct = \
                np.unique(asrs_yr_mth['tracon_code'].astype(str), return_index=True, \
                return_counts=True)

        for trcn_code_elem_idx, trcn_code in enumerate(trcn_codes):
            if pd.isna(trcn_code):
                continue
            trcn_start = trcn_codes_idx[trcn_code_elem_idx]
            trcn_end = trcn_start + trcn_codes_ct[trcn_code_elem_idx]
            asrs = asrs_yr_mth.iloc[trcn_start:trcn_end].copy()

            # this creates the string id of the given tracon_month
            index_id = f'{trcn_code} {year}/{mth}'

            other_info = {}
            if not unq_only:
                # asrs[f'{col}_wc'] = asrs.apply(lambda x: \
                #         preprocess_helper.convert_to_words(x, col).shape[0], axis=1)
                asrs[f'{col}_wc'] = asrs[col].apply(preprocess_helper.count_words, axis=1)

                # fill in additional columns
                any_col_has_multiple_reports = get_selector_for_mult_reports(asrs, other_info)

                other_info['avg_code_per_obs'] = asrs['num_code_per_obs'].mean()
                other_info['num_total_idents'] = get_total_num_idents(asrs, other_info, unique_idents)
                other_info['num_multiple_reports'] = any_col_has_multiple_reports.sum()
                other_info['num_observations'] = asrs.shape[0]
                other_info['num_callbacks'] = asrs['contains_callback'].sum()
                other_info[f'{abrev_col}_wc'] = asrs[f'{col}_wc'].sum()
                other_info[f'{abrev_col}_avg_wc'] = asrs[f'{col}_wc'].mean()
                other_info['tracon_code'] = trcn_code

            # this is redundant (occurs in preprocess_helper.py)
            asrs[col] = asrs[col].str.lower()

            # this creates a collections.Counter object that counts the number of times each word
            # showed up within the given tracon_month, then saved to index_to_counter
            split = np.sum(asrs[col].apply(preprocess_helper.convert_to_ctr).values)
            index_to_counter[index_id] = split

            # index_to_counter[index_id] = Counter(np.hstack(split.values).flatten())
            if not unq_only:
                index_to_other_info[index_id] = pd.Series(other_info)
    return index_to_counter, index_to_other_info

def generate_tracon_month_ctr(col, abrev_set, key_ctr, prefix, unq_only=False):
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
    # converts full column name to shortened column name
    abrev_col = preprocess_helper.ABREV_COL_DICT[col]
    pos_nonword_df = pd.DataFrame.from_dict({key: convert_ctr_to_series(ctr, abrev_set, abrev_col, \
            unq_only) for key, ctr in key_ctr}, orient='index')
    pos_nonword_df = pos_nonword_df.add_prefix(prefix)
    return pos_nonword_df

def load_dictionary_sets(total_cts):
    """
    Creates sets for different categorizations of words (pos_nwrd, all_abrevs, aviation_dicts)
    """
    aviation_dicts = preprocess_helper.load_dictionaries()

    # set of abbreviations to calculate counts
    all_abrevs = set(total_cts.loc[total_cts['abrev'] == 1, 'acronym'])
    pos_nonword_abrevs = set(total_cts.loc[total_cts['tag'] == 'pos_nonword', 'acronym'])

    # the list of set of acronyms for each dictionary. this is used to create counts for each
    # tracon month (aligned with aviation dicts)
    df_sets = [set(x['acronym']).intersection(all_abrevs) for x in aviation_dicts.values()]
    return all_abrevs, pos_nonword_abrevs, df_sets

def postprocess_unq_all_df(unq_all_df):
    unq_all_df['year'] = unq_all_df.index.map(get_year)
    unq_all_df['month'] = unq_all_df.index.map(get_month)
    return unq_all_df

def calculate_all_unq(key_ctr, word_set, col, prefix):
    abrev_col = preprocess_helper.ABREV_COL_DICT[col]
    yr_mth_to_ctrs = {}
    for key, ctr in key_ctr:
        index = key.split()[1]
        yr_mth_to_ctrs[index] = yr_mth_to_ctrs.get(index, []) + [ctr]

    for yr_mth in yr_mth_to_ctrs:
        fin_res = convert_ctr_to_series(np.sum(yr_mth_to_ctrs[yr_mth]), abrev_set=word_set, \
                unq_only=True, abrev_col=abrev_col)
        yr_mth_to_ctrs[yr_mth] = fin_res
    # create df and postprocess
    unq_all_df = pd.DataFrame.from_dict(yr_mth_to_ctrs, orient='index')
    unq_all_df = unq_all_df.add_prefix(prefix)
    unq_all_df = postprocess_unq_all_df(unq_all_df).reset_index().drop('index', axis=1)
    return unq_all_df

def generate_ctr_df(total_cts, index_to_counter, index_to_other_info, col, unq_only=False):
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
    all_abrevs, pos_nonword_abrevs, df_sets = load_dictionary_sets(total_cts)
    aviation_dicts = preprocess_helper.load_dictionaries()

    other_info = pd.DataFrame.from_dict(index_to_other_info, orient='index')
    key_ctr = list(index_to_counter.items())

    pos_nonword_df = generate_tracon_month_ctr(col, pos_nonword_abrevs, key_ctr, "pos_nwrd_", unq_only)
    all_df = generate_tracon_month_ctr(col, all_abrevs, key_ctr, "abrvs_no_ovrcnt_", unq_only)

    all_dfs = [pos_nonword_df, all_df, other_info]
    # count the number of times words in each dictionary shows up in each tracon_month
    for dict_idx, dict_name in enumerate(aviation_dicts.keys()):
        cts = generate_tracon_month_ctr(col, df_sets[dict_idx], key_ctr, f"{dict_name}_", unq_only)
        cts.to_csv(f'results/tracon_month_{col}_{dict_name}.csv')
        all_dfs.append(cts)

    all_dfs = pd.concat(all_dfs, axis=1)
    all_dfs.index = all_dfs.index.rename('tracon_month')

    all_dfs = add_dates(all_dfs)

    if unq_only:
        all_abrevs_unq = calculate_all_unq(key_ctr, all_abrevs, col, "abrvs_no_ovrcnt_")
        all_posnwrd_unq = calculate_all_unq(key_ctr, pos_nonword_abrevs, col, "pos_nwrd_")
        new_all_dfs = [all_abrevs_unq, all_posnwrd_unq]
        for dict_idx, dict_name in enumerate(aviation_dicts.keys()):
            new_all_dfs.append(calculate_all_unq(key_ctr, df_sets[dict_idx], col, f"{dict_name}_"))
        unq_all = reduce(lambda x, y: pd.merge(x, y, on=['year', 'month']), new_all_dfs)
        unq_all = unq_all.add_suffix("_all")
        unq_all.rename({'year_all': 'year', 'month_all': 'month'}, inplace=True, axis=1)
        fin = pd.merge(all_dfs.reset_index(), unq_all, on=['year', 'month']).set_index(\
                'tracon_month')
        orig_cols = [col for col in fin.columns if col in fin.columns and f"{col}_all" in fin.columns]
        for unq_col in orig_cols:
            fin[f"{unq_col}_out"] = fin[f"{unq_col}_all"] - fin[f"{unq_col}"]
        return fin
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
            and col not in ['year', 'month', 'num_observations', 'tracon_code']]

    num_obs = all_dfs[sel + ['num_observations']].drop_duplicates(sel)
    num_obs = num_obs.groupby(['year', 'month']).sum().reset_index()
    for _, row in num_obs.iterrows():
        yr_mth_sel = (all_dfs['year'] == row['year']) & (all_dfs['month'] == row['month'])
        all_dfs.loc[yr_mth_sel, 'num_observations_all'] = row['num_observations']
    all_dfs['num_observations_out'] = all_dfs['num_observations_all'] - all_dfs['num_observations']

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

    # for col in cols:
    #     total_cts = load_total_cts(col)
    #
    #     # create dictionaries mapping from tracon_month to cts
    #     index_to_counter, index_to_other_info = create_index_dicts(all_pds, col)
    #
    #     # generate count dataframes
    #     all_dfs = generate_ctr_df(total_cts, index_to_counter, index_to_other_info, col)
    #     all_dfs = analyze_wc(all_dfs)
    #
    #     # add rows for missing tracon_months
    #     all_dfs = add_missing_rows(all_dfs)
    #
    #     # post-process and save
    #     all_dfs.drop(['year', 'month'], axis=1, inplace=True)
    #     all_dfs.to_csv(f'results/tracon_month_{col}.csv', index=True)

    # calculate unique counts
    for col in cols:
        total_cts = load_total_cts(col)

        for month_range in MONTH_RGS:
            index_to_counter, index_to_other_info = create_index_dicts(all_pds, col, \
                    month_range, lag=1, unq_only=True)
            all_dfs = generate_ctr_df(total_cts, index_to_counter, index_to_other_info, col, \
                    unq_only=True)

            # add rows for missing tracons
            all_dfs = add_missing_rows(all_dfs)

            all_dfs.drop(['year', 'month'], axis=1, inplace=True)
            all_dfs.to_csv(f'results/tracon_month_unq_{col}_{month_range}mon.csv', index=True)

if __name__ == "__main__":
    main()
