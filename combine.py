"""
Combines ASRS data with FAA/NTSB, LIWC and Doc2Vec datasets
"""
import argparse
import os
import re
from itertools import product

from IPython import embed
from tqdm import tqdm

import pandas as pd
import numpy as np

from helper import get_tracon, get_year, get_month, fill_with_empty

ifr_vfr_dict = {
    'itinerant': 'itnr',
    'general': 'gen',
    'overflight': 'ovrflt'
}
all_cols = ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']
faa_ntsb_cols = ['ntsb_incidents', 'ntsb_accidents', 'faa_incidents']
num_months = [1, 3, 6, 12]

if not os.path.exists('results/final/'):
    os.makedirs('results/final/')

def num_months_between(month1, year1, month2, year2):
    return (year2 - year1) * 12 + month2 - month1

def generate_compare_npy(year1, month1, num_months=1):
    def inner_func(arr):
        year2, month2 = arr
        n_m = num_months_between(month2, year2, month1, year1)
        return n_m > 0 and n_m <= num_months
    return inner_func

def year_month_indices(yr_mth, yr_mth_idx, yr_mth_cts, year1, month1, num_months=1):
    func = generate_compare_npy(year1, month1, num_months)
    sel = np.apply_along_axis(func, 1, yr_mth)

    idx_of_sel = sel.nonzero()[0]
    if len(idx_of_sel) == 0:
        return []
    start = yr_mth_idx[idx_of_sel[0]]
    end = yr_mth_idx[idx_of_sel[-1]] + yr_mth_cts[idx_of_sel[-1]]
    return list(range(start, end))


def rename_cols_dict(output_df, month_range_str, skip_cols=[]):
    """
    Create a renaming dictionary on the given output dataframe
    @param: output_df (pd.DataFrame) output dataframe
    @param: month_range_str(str) '1m', '3m', ... '12m', 'atime'
        Represents what time range you are looking at
    @param: skip_cols (list of str) list of columns that we will not add
        the month_range_str to
    @returns: dict[orig_name] -> new_name
    """
    except_cols = set(['year', 'month', 'tracon_key'] + skip_cols)
    rename_dict = {'airport_code': 'tracon_key'}
    for col in output_df.columns:
        if col == 'airport_code':
            continue
        new_col = col
        if 'IFR' in col or 'VFR' in col:
            split_col = col.lower().split("\t")
            start_str = ''.join([ifr_vfr_dict.get(x, x) for x in split_col[1].split()])
            end_str = '_'.join([ifr_vfr_dict.get(x, x) for x in split_col[0].split()])
            new_col = f'{start_str}_{end_str}'
        elif 'Local' in col:
            split_col = col.lower().split()
            new_col = f'{split_col[1]}_{split_col[0]}'
        if new_col not in except_cols and not new_col.endswith(month_range_str):
            new_col += "_" + month_range_str

        new_col = new_col.replace(" ", "_").replace("narrative", "narr").lower()
        new_col = new_col.replace("\t", "_")
        rename_dict[col] = new_col
    return rename_dict

def rename_cols(output_df, month_range_str, skip_cols=[]):
    """
    Renames the columns of the  output dataframe
    @param: output_df (pd.DataFrame) output dataframe
    @param: month_range_str(str) '1m', '3m', ... '12m', 'am'
        Represents what time range you are looking at
    @param: skip_cols (list of str) which columns to skip
    @returns: renamed dataframe
    """
    return output_df.rename(rename_cols_dict(output_df, month_range_str, skip_cols), axis=1)

def tuple_to_column(tuple_input):
    """
    Returns a string representation of the tuple_input by joining on underscores,
    however some of the elements of the tuple may be empty so some postprocesing is
    performed
    @param: tuple_input (tuple of str) the strings you want to join
    @returns joined_input (str)
    """
    fin_str = "_".join(tuple_input)
    return re.sub('_{2,}', '_', fin_str)

def generate_cartesian_prod_columns(list_args):
    """
    Helper function that creates a list of columns given a list of lists, each list
    representing the possible parts of the str
        Ex: list_args = [['a', 'b'], ['c', 'd']] -> ['a_c', 'a_d', 'b_c', 'b_d']
    @param: list_args (list of lists)
    @returns list of column names
    """
    return list(map(tuple_to_column, product(*list_args)))

def reorder_cols(final_df):
    """
    This reorders the columns of the final output
    @param: final_df (pd.DataFrame) final output
    @returns: re-ordered df (pd.DataFrame)
    """
    cols = ['tracon_key', 'year', 'month', 'ntsb_accidents', \
            'ntsb_incidents', 'faa_incidents', 'faa_ntsb_overlap', \
            'ntsb_faa_incidents_total', 'ntsb_faa_incidents_total_nodups', \
            'state', 'region', 'ddso_service_area', 'class', 'tower_operations', \
            'airport_operations', 'total_operations', 'faa_ntsb_overlap']
    # volume columns
    flight_types = ['aircarrier', 'airtaxi', 'genaviation', 'military', 'total']
    itr_ovr = ['itnr', 'ovrflt']
    ifr_vfr = ['ifr', 'vfr']

    vol_cols = generate_cartesian_prod_columns([flight_types, ifr_vfr, itr_ovr])
    vol_cols = vol_cols[:10] + ['military_local', 'civil_local', 'total_local'] + \
            vol_cols[10:]
    vol_cols = [x.replace("_1m", "") for x in vol_cols]
    cols = cols + vol_cols

    # wc columns
    text_columns = ['narr', 'syn', 'call', 'narrsyn', 'all']
    word_c = ['avg_wc', 'wc']
    sel = ['', 'all', 'out', 'prop']
    time_windows = ['1m', '3m', '6m', '12m', 'atime']
    wc_cols = generate_cartesian_prod_columns([text_columns, word_c, sel, time_windows])
    cols = cols + wc_cols

    # trcn columns (d2v)
    cols = cols + [col for col in final_df.columns if 'trcn' in col]

    # some other ct columns
    ct_cols = ['pos_nwrd', 'pos_nwrd_unq', 'abrvs_no_ovrcnt', 'abrevs_no_ovrcnt_unq']
    ct_cols = generate_cartesian_prod_columns([ct_cols, text_columns, time_windows])
    cols = cols + ct_cols

    # aviation dicts
    aviation = ['nasa', 'faa', 'casa', 'iata', 'hand', 'hand2']
    unique = ['unq', '']

    aviation_cols = generate_cartesian_prod_columns([aviation, unique, text_columns, time_windows])
    cols = cols + aviation_cols

    # liwc cols
    liwc = ['liwc']
    flfrm = ['', 'flfrm']
    sel = ['ct', 'prop']
    liwc_cat = set()
    for col in final_df.columns:
        for time_w in time_windows:
            for text_col in text_columns:
                liwc_pat = re.compile('liwc_([a-z]{1,})_' + f'{text_col}_ct_{time_w}')
                pat_res = liwc_pat.match(col)
                if pat_res is not None:
                    liwc_cat.add(pat_res.group(1))
    liwc_cat = list(liwc_cat)
    liwc_cols = generate_cartesian_prod_columns(\
            [liwc, liwc_cat, flfrm, text_columns, sel, time_windows])
    cols = cols + liwc_cols

    # ident_ct cols
    ident_ct_cols = [x for x in final_df.columns if 'ident_ct' in x]
    cols = cols + ident_ct_cols

    # deal with identical columns
    common_cols = ['num_total_idents', 'num_multiple_reports', \
            'num_observations', 'num_callbacks']
    common_col_dict = {}
    only_once = []
    for col in common_cols:
        only_once.append(col)
        for time_w in time_windows:
            common_col_dict[f'{col}_{time_w}'] = col
    for col in vol_cols:
        common_col_dict[col + "_1m"] = col
    cols = cols + only_once

    final_df.rename(common_col_dict, axis=1, inplace=True)

    final_df = final_df.loc[:, ~final_df.columns.duplicated()]
    return final_df.loc[:, [x for x in cols if x in final_df.columns]]

def load_ame():
    """
    Loads airport_month_events (or combined_vol_incident.csv), which was generated by
    flight_vol.py. Also known as incident/accident ds (inc_acc_ds). Only selects rows
    with airport codes in the top50 iata.
    @returns: airport_month_events (pd.DataFrame) of inc_acc_ds.
    """
    airport_month_events = pd.read_csv('results/combined_vol_incident.csv')
    top_50_iata = \
            set(pd.read_excel('datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    airport_month_events['airport_code'] = airport_month_events['airport_code'].astype(str)
    airport_month_events = airport_month_events.loc[\
            airport_month_events['airport_code'].apply(lambda x: x.split()[0] in top_50_iata)]

    airport_month_events = fill_with_empty(airport_month_events, 'airport_code')
    return airport_month_events

def preprocess_liwc(liwc_df):
    """
    Processes LIWC dataframe (generated from asrs_analysis/liwc_analysis.py) by extracting
    tracon/month/year inofmration from its index.
    @param: liwc_df (pd.DataFrame) of liwc counts from asrs dataset
    @returns: liwc_df (pd.DataFrame) with extracted columns
    """
    liwc_df['tracon'] = pd.Series(liwc_df.index.map(get_tracon), index=liwc_df.index)
    liwc_df['month'] = pd.Series(liwc_df.index.map(get_month), index=liwc_df.index)
    liwc_df['year'] = pd.Series(liwc_df.index.map(get_year), index=liwc_df.index)
    liwc_df = liwc_df.reset_index().rename({'index':'tracon_month'}, axis=1)
    return liwc_df

def preprocess_asrs(asrs):
    """
    Processes ASRS dataframe by extracting tracon/month/year information from its index.
    @paam: asrs (pd.DataFrame) asrs dataset
    @returns: asrs (pd.DataFrame) w/extracted columns
    """
    asrs['tracon'] = asrs.index.map(get_tracon)
    asrs['month'] = asrs.index.map(get_month)
    asrs['year'] = asrs.index.map(get_year)

    asrs = asrs.reset_index().rename({'index':'tracon_month'}, axis=1)
    return asrs

def combine_asrs_liwc(col):
    """
    Combines ASRS dataset with LIWC counts dataset by joining on tracon_month. Extracts
    dataset from asrs_analysis subdirectory.
    @param: col (str) column that we are analyzing
    @returns: asrs_orig (pd.DataFrame) of merged asrs/liwc dataset
    """
    asrs = pd.read_csv(f'asrs_analysis/results/tracon_month_{col}.csv', index_col=0)
    asrs = preprocess_asrs(asrs)

    # liwc counts
    liwc_df = pd.read_csv(f'asrs_analysis/results/liwc_tracon_month_{col}_counts.csv', \
            index_col=0, header=[1])
    liwc_df = preprocess_liwc(liwc_df)

    asrs_orig = asrs.merge(liwc_df.drop(['tracon', 'month', 'year'], axis=1),\
            on='tracon_month', how='outer')
    asrs_orig.sort_values(['year', 'month', 'tracon'], inplace=True)
    return asrs_orig

def generate_cached_dicts(airport_month_events, asrs, n_month, yr_mth_info):
    """
    For a given month_range (1/3/6/12 months), we calculate cached dictionaries in order
    to combine the datasets as quickly as possible. We define the dictionaries below.
    @param: airport_month_events (pd.DataFrame) of inc_acc_ds
    @param: asrs (pd.DataFrame) dataframe consisting of ASRS info + LIWC info
    @param: n_month (int) month range
    @param: yr_mth_info (list of np.ndarray). This is the result of calling
        np.unique(asrs[['year', ',month']], return_idx=True, return_cts=True)
        yr_mth_info[0] = np.ndarray of all unique combinations of year/month (shape (n, 2))
            where the first column indexes into year, and the second into month
        yr_mth_info[1] = np.ndarray of indices of the first indices in which,
            that particular year/month combination occurs in asrs dataset. (shape (n,)),
            notice that the 0th dimension is equivalent to that of yr_mth_info[0],
            so yr_mth_info[1][idx] = index at which yr_mth_info[0] occurs in ASRS
        yr_mth_info[2] = np.ndarray of indices of the number of times in which,
            that particular year/month combination occurs in asrs dataset. (shape (n,)),
            notice that the 0th dimension is equivalent to that of yr_mth_info[0],
            so yr_mth_info[2][idx] = number of times yr_mth_info[0] occurs in ASRS
            Assume that asrs is sorted by year/month then tracon
    @returns: tracon_month_dict (dict[date] -> subset of ASRS within n_month months of
        that particular date) dict['8 2011'] = rows of ASRS with tracon_month that are
        within n_months of that date (but not including 08/2011)
    @returns: unique_info (dict[date] -> list of np.ndarray), result of calling
        np.unique(tracon_month_dict[date]['tracon'].values..., return_index=True, \
                return_counts=True), same format as yr_mth_info, see above.
    """
    yr_mth, yr_mth_idx, yr_mth_ct = yr_mth_info
    # this creates a dictionary from year/month -> pd.DataFrame of all the rows in
    # the ASRS dataset within the month range (utilizing n_month)
    # ex.: January 2011, w/ n_month = 1 -> pd.DataFrame of all rows in ASRS in December 2010
    tracon_month_dict, unique_info = {}, {}
    month_year_df = airport_month_events[['month', 'year']].drop_duplicates()
    for _, date_row in tqdm(month_year_df.iterrows(), total=month_year_df.shape[0], desc=\
            f"Creating year/month dictionary {n_month}mon"):
        month, year = int(date_row['month']), int(date_row['year'])
        code = ' '.join([str(month), str(year)])

        yr_mth_sel_idx = year_month_indices(yr_mth, yr_mth_idx, yr_mth_ct, int(year), int(month), \
                num_months=n_month)
        tracon_month_dict[code] = asrs.iloc[yr_mth_sel_idx, :].copy()
        unique_info[code] = np.unique(tracon_month_dict[code]['tracon'].values.astype(str), \
                return_index=True, return_counts=True)
    return tracon_month_dict, unique_info

def compute_tracon_indices(unq_info, trcn_code):
    """
    This computes the indices of a dataframe with a particular trcn_code. Assumes
    that the dataframe from which unq_info was generated is sorted by trcn_code.
    @param: unq_info (list of np.ndarray), this is generated by generate_cached_dicts
        (the unique_info.values()), and is the result of calling np.unique (see above).
    @param: trcn_code (str)
    @returns: list of indices. Each index indicates that that index in a given dataframe
        has a trcn_code as given by the parameters.
    """
    unq_codes, unq_idx, unq_cts = unq_info

    idx_of_code = np.searchsorted(unq_codes, trcn_code)
    if idx_of_code >= unq_codes.shape[0] or unq_codes[idx_of_code] != trcn_code:
        same_tracon = []
    else:
        start, end = unq_idx[idx_of_code], unq_idx[idx_of_code] + unq_cts[idx_of_code]
        same_tracon = list(range(start, end))
    return same_tracon

def generate_final_row(row, searched, d2v_tm, month, year, month_idx, n_month, cumulative_index):
    """
    This combines one row of the ASRS dataset with one row of the d2v dataset for a given
    month_range and month/year. We ensure that if the month/year is smaller than n_months away
    from the beginning (January 1988) to fill the d2v portion of the final_row with nans,
    as otherwise we would not be comparing apples to apples.
    @param: row (pd.Series) a row from airport_month_events or the inc_acc_ds
    @param: searched (pd.DataFrame), the portion of the ASRS/LIWC dataset that are within
        n_months of the date described by month/year (the parameters). This is summed together
        to calculate our final row
    @param: d2v_tm (pd.DataFrame), the dataframe for the cos_sim results for this particular
        month_range and column
    @param: month (int) describes the current month of the row from the inc_acc_ds that we
        are analyzing
    @param: year (int) describes the current year of the row from the inc_acc_ds that we are
        analyzing
    @param: month_idx (int), which month_range we are utilizing. This indexes into num_months
    @param: n_month (int), the month_range value
    @param: cumulative_index (pd.Index) is an iterable of what columns exist in ASRS/LIWC df
    @returns: pd.Series with all relevant columns that are linked with the month/year combination
        from row (of inc_acc_ds/airport_month_events). Includes average d2v numbers, LIWC numbers,
        all other calculations made in the whole pipeline from d2v/liwc/abbrev/etc.
    """
    d2v_index = d2v_tm.columns
    d2v_code = f'{row["airport_code"]} {int(year)}/{int(month)}'
    assert d2v_code in d2v_tm.index

    tr_yr_mon = row[['airport_code', 'year', 'month']]

    num_month_from_start = num_months_between(1, 1988, month, year)
    concat_series = []

    # after all month_ranges are generated, we concatenate the dataframes together
    # however, the airport_month_events columns should not be duplicated, so we only
    # add the row if month_idx == 0 (the first dataframe)
    if month_idx == 0:
        concat_series.append(row)
    else:
        concat_series.append(tr_yr_mon)

    # select asrs rows, if it's beginning or no rows found in asrs, then add nans
    if (num_month_from_start >= 0 and num_month_from_start < n_month and n_month != np.inf) or \
            searched.shape[0] == 0:
        concat_series.append(pd.Series(index=cumulative_index, dtype='float64'))
    else:
        cumulative = searched.drop(['tracon_month', 'tracon', 'year', 'month'], axis=1).sum()
        concat_series.append(cumulative)

    # select d2v rows, if it's the beginning, remove those rows
    if ((num_month_from_start >= 0) and (num_month_from_start < n_month) and (n_month != np.inf)):
        concat_series.append(pd.Series(index=d2v_index, dtype='float64'))
    else:
        concat_series.append(d2v_tm.loc[d2v_code])
    return pd.concat(concat_series, axis=0)

def generate_final_ds_col_month_range(airport_month_events, asrs, d2v_tm, tracon_month_dict, \
        unique_info, month_idx, n_month):
    """
    This generates the final dataset (combining elements from ASRS/LIWC/D2V/ABBREV) for
    a given month_range.
    @param: airport_month_events (pd.DataFrame) is the inc_acc_ds w/volume data.
    @param: asrs (pd.DataFrame) contains ASRS data and LIWC data
    @param: d2v_tm (pd.DataFrame) contains D2V data generated from asrs_analysis/cos_sim.py
    @param: unique_info (dict[date] -> list of np.ndarray), result of calling
        np.unique(tracon_month_dict[date]['tracon'].values..., return_index=True, \
                return_counts=True), same format as yr_mth_info, see above.
        This is generated from generate_cached_dicts (also see combined_col_month_range)
    @param: month_idx (int), which month_range we are utilizing. This indexes into num_months
    @param: n_month (int), the month_range value
    @returns: final_dataset (pd.DataFrame) with elements from all datasets
    """
    final_rows, asrs_covered_ind = [], set()
    ctr = 0
    cumulative_index = asrs.columns.drop(['tracon_month', 'tracon', 'year', 'month'])
    for _, row in tqdm(airport_month_events.iterrows(), total=airport_month_events.shape[0], \
            desc=f"Combining ASRS {n_month}mon"):
        month, year = float(row['month']), float(row['year'])
        code = ' '.join([str(int(month)), str(int(year))])
        if code in tracon_month_dict:
            searched = tracon_month_dict[code]

            trcn_code = row['airport_code']
            # get indices of dataframe corresponding to our tracon month
            same_tracon = compute_tracon_indices(unique_info[code], trcn_code)
            searched = searched.iloc[same_tracon, :]

            asrs_covered_ind.update(searched.index)

            # row, month, year, month_idx, n_month, cumulative_index, searched
            final_rows.append(generate_final_row(row, searched, d2v_tm, month, year, \
                    month_idx, n_month, cumulative_index))
        else:
            ctr += 1
    print('ctr', ctr)
    print('% ASRS covered', len(asrs_covered_ind) / asrs.shape[0])
    print('% incident covered', len(asrs_covered_ind) / airport_month_events.shape[0])
    # generate dataset and perform some preprocessing
    res = pd.DataFrame.from_dict({idx: row for idx, row in enumerate(final_rows)}, \
            orient='index')
    return res

def postprocess_final_ds(res, month_range_str, ame_cols):
    """
    Postprocesses final dataset by renaming columns, filling nas with 0s, and
    dropping duplicated columns.
    @param: res (pd.DataFrame) final dataset generated from generate_final_ds_col_month_range
    @param: month_range_str (str) str representing what month_range we are utilizing
        see combine_col_month_range
    @param: ame_cols (list of str), list of columns from airport_month_events or inc_acc_ds
    @returns: processed res (pd.DataFrame)
    """
    res = rename_cols(res, month_range_str, skip_cols=ame_cols)

    for fn_col in faa_ntsb_cols:
        if fn_col in res.columns:
            res.loc[res[fn_col].isna(), fn_col] = 0

    res = res.loc[:, ~res.columns.duplicated()]
    res.set_index(['tracon_key', 'year', 'month'], inplace=True)
    return res

def combine_col_month_range(month_idx, n_month, yr_mth_info, asrs_orig, airport_month_events, col):
    """
    Given a column and a month_range, we combine the doc2vec, liwc, asrs, and airport_month_events
    datasets together.
    @param: month_idx (int), which month_range we are utilizing. This indexes into num_months
    @param: n_month (int) month range
    @param: yr_mth_info (list of np.ndarray). This is the result of calling
        np.unique(asrs[['year', ',month']], return_idx=True, return_cts=True) (see main)
        also see generate_cached_dicts for more information
    @param: asrs_orig (pd.DataFrame) dataframe including liwc/asrs datasets
    @param: airport_month_events (pd.DataFrame) the dataset that includes inc_acc_ds.
    @param: col (str) which column we are analyzing
    @returns: res (pd.DataFrame) processed dataframe including information from all datasets
    """
    ame_cols = list(airport_month_events.columns)

    # helper info
    month_range_str = f'{n_month}m'
    if n_month == np.inf:
        month_range_str = 'atime'

    # load datasets
    asrs = asrs_orig.copy()
    d2v_tm = pd.read_csv(f'./asrs_analysis/results/d2v_tracon_month_{col}_{n_month}mon.csv', \
            index_col=0)
    d2v_tm.index = d2v_tm.index.rename('tracon_month')

    tracon_month_dict, unique_info = generate_cached_dicts(airport_month_events, \
            asrs, n_month, yr_mth_info)
    
    res = generate_final_ds_col_month_range(airport_month_events, asrs, d2v_tm, \
            tracon_month_dict, unique_info, month_idx, n_month)
    res = postprocess_final_ds(res, month_range_str, ame_cols)
    res.to_csv(f'results/final/{col}_{month_range_str}.csv')
    return res

def ensure_multi_index(all_res):
    """
    Forces each element of all_res to have a multi-index consisting of tracon_key/year/month.
    @param: all_res (list of pd.DataFrame)
    @returns: modified all_res
    """
    for idx in range(len(all_res)):
        if not isinstance(all_res[idx].index, pd.MultiIndex):
            all_res[idx] = all_res[idx].reset_index().set_index(['tracon_key', 'year', 'month'])
    return all_res

def main():
    airport_month_events = load_ame()

    all_res = []
    for col in all_cols:
        asrs_orig = combine_asrs_liwc(col)

        yr_mth_info = np.unique(asrs_orig[['year', 'month']].values.astype(int), axis=0, \
                return_index=True, return_counts=True)

        for month_idx, n_month in enumerate(num_months):
            res = combine_col_month_range(month_idx, n_month, yr_mth_info, asrs_orig, \
                    airport_month_events, col)
            all_res.append(res)
            embed()

    all_res = ensure_multi_index(all_res)
    all_res = pd.concat(all_res, ignore_index=False, axis=1, copy=False)
    all_res = reorder_cols(all_res)
    all_res.to_csv('results/final_dataset.csv')

    coverage = all_res.isna().sum()
    coverage['total rows'] = all_res.shape[0]
    coverage.to_csv('results/final_coverage.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine ASRS data w/FAA/NTSB/LIWC/D2V.')
    parser.add_argument('-t', action='store_true')
    args = parser.parse_args()

    test = args.t
    SKIP_EMPTY = False
    main()
