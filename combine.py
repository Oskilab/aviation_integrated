import pandas as pd, numpy as np, re
from IPython import embed
from tqdm import tqdm
from itertools import product
"""
Combines ASRS data with FAA/NTSB, LIWC and Doc2Vec datasets
"""
ifr_vfr_dict = {
    'itinerant': 'itnr',
    'general': 'gen',
    'overflight': 'ovrflt'
}

num_time_periods = (2020 - 1988) * 12
def rename_cols_dict(pd, month_range_str, skip_cols = []):
    except_cols = set(['year', 'month', 'tracon_key'] + skip_cols)
    rename_dict = {'airport_code': 'tracon_key'}
    for col in pd.columns:
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

def rename_cols(pd, month_range_str, skip_cols = []):
    return pd.rename(rename_cols_dict(pd, month_range_str, skip_cols), axis = 1)

def reorder_cols(df):
    import re
    def tuple_to_column(tuple_input):
        tuple_input = tuple_input[::-1]
        fin_str = "_".join(tuple_input)
        return re.sub('_{2,}', '_', fin_str)
    # currently missing airport_name after tracon_key
    # missing dataset after faa_incidents
    cols = ['tracon_key', 'year', 'month', 'ntsb_accidents', \
            'ntsb_incidents', 'faa_incidents', 'state', 'region', \
            'ddso_service_area', 'class', 'tower_operations', 'airport_operations', \
            'total_operations']
    # volume columns
    flight_types = ['aircarrier', 'airtaxi', 'genaviation', 'military', 'total']
    itr_ovr = ['itnr', 'ovrflt']
    ifr_vfr = ['ifr', 'vfr']
    end_str=  ["1m"]

    vol_cols = list(map(\
            # tuple_to_column, product(flight_types, ifr_vfr, itr_ovr, end_str)\
            tuple_to_column, product(end_str, itr_ovr, ifr_vfr, flight_types)\
            ))
    vol_cols = vol_cols[:10] + ['military_local', 'civil_local', 'total_local'] + \
            vol_cols[10:]
    vol_cols = [x.replace("_1m", "") for x in vol_cols]
    cols = cols + vol_cols

    # wc columns
    text_columns = ['narr', 'syn', 'call', 'narrsyn', 'all']
    wc = ['avg_wc', 'wc']
    sel = ['', 'all', 'out', 'prop']
    time_windows = ['1m', '3m', '6m', '12m', 'atime']
    wc_cols = list(map(\
            # tuple_to_column, product(text_columns, wc, sel, time_windows)\
            tuple_to_column, product(time_windows, sel, wc, text_columns)\
            ))
    cols = cols + wc_cols

    # trcn columns (d2v)
    cols = cols + [col for col in df.columns if 'trcn' in col]

    # some other ct columns
    ct_cols = ['pos_nwrd', 'pos_nwrd_unq', 'abrvs_no_ovrcnt', 'abrvs_no_ovrcnt_unq']
    ct_cols = list(map(\
            # tuple_to_column, product(ct_cols, text_columns, time_windows)\
            tuple_to_column, product(time_windows, text_columns, ct_cols)\
            ))
    cols = cols + ct_cols

    # aviation dicts
    aviation = ['nasa', 'faa', 'casa', 'iata', 'hand', 'hand2']
    unique = ['unq', '']
    sel = ['', 'prop']
    aviation_cols = list(map(\
            # tuple_to_column, product(aviation, unique, sel, text_columns, time_windows)\
            tuple_to_column, product(time_windows, text_columns, sel, unique, aviation)\
            ))
    cols = cols + aviation_cols

    # liwc cols
    liwc = ['liwc']
    liwc_cat = set()
    for col in df.columns:
        for tw in time_windows:
            for text_col in text_columns:
                liwc_pat = re.compile('liwc_([a-z]{1,})_' + f'{text_col}_ct_{time_windows}')
                pat_res = liwc_pat.match(col)
                if pat_res is not None:
                    liwc_cat.add(pat_res.group(1))
    liwc_cat = list(liwc_cat)
    flfrm = ['', 'flfrm']
    sel = ['ct', 'prop']
    liwc_cols = list(map(\
            # tuple_to_column, product(liwc, liwc_cat, flfrm, text_columns, sel, time_windows)\
            tuple_to_column, product(time_windows, sel, text_columns, flfrm, liwc_cat, liwc)\
            ))
    cols = cols + liwc_cols

    # ident_ct cols
    ident_ct_cols = [x for x in df.columns if 'ident_ct' in x]
    cols = cols + ident_ct_cols

    # deal with identical columns
    common_cols = ['num_total_idents', 'num_multiple_reports', \
            'num_observations', 'num_callbacks']
    common_col_dict = {}
    only_once = []
    for col in common_cols:
        only_once.append(col)
        for tw in time_windows:
            common_col_dict[f'{col}_{tw}'] = col
    for x in vol_cols:
        common_col_dict[x + "_1m"] = x
    cols = cols + only_once

    df.rename(common_col_dict, axis = 1, inplace = True)

    df = df.loc[:,~df.columns.duplicated()]
    print('dropped columns', set(df.columns) - set([x for x in cols if x in df.columns]))
    return df.loc[:, [x for x in cols if x in df.columns]]


# ntsb/faa incident dataset + volume
airport_month_events = pd.read_csv('results/combined_vol_incident.csv', index_col = 0)
ame_cols = list(airport_month_events.columns)

# for missing tracon codes
unique_codes = airport_month_events['airport_code'].unique()
all_combs = set()
for idx, row in airport_month_events[['airport_code', 'year', 'month']].drop_duplicates() \
        .iterrows():
    code, mon, year = row['airport_code'], row['month'], row['year']
    all_combs.add((code, mon, year))

missing_tracons = None
all_res = []
# for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
for col in ['narrative']:
    print('col', col)
    # asrs
    tracon_nar = pd.read_csv(f'asrs_analysis/results/tracon_month_{col}.csv', index_col = 0)

    # liwc counts
    liwc_df = pd.read_csv(f'asrs_analysis/results/liwc_tracon_month_{col}_counts.csv', \
            index_col = 0, header = [1])

    def get_tracon(x):
        split_x = str(x).split()
        if len(split_x) > 2:
            return " ".join(split_x[:-1])
        else:
            return split_x[0]

    def get_year(x):
        try:
            return int(float(str(x).split()[-1].split("/")[0]))
        except ValueError:
            return np.nan

    def get_month(x):
        try:
            return int(float(str(x).split()[-1].split("/")[1]))
        except ValueError:
            return np.nan


    # preprocess liwc_df
    liwc_df['tracon'] = pd.Series(liwc_df.index.map(get_tracon), index = liwc_df.index)
    liwc_df['month'] = pd.Series(liwc_df.index.map(get_month), index = liwc_df.index)
    liwc_df['year'] = pd.Series(liwc_df.index.map(get_year), index = liwc_df.index)
    liwc_df = liwc_df.reset_index().rename({'index':'tracon_month'}, axis = 1)

    # preprocess asrs
    asrs = tracon_nar.reset_index()
    asrs['tracon'] = pd.Series(tracon_nar.index.map(get_tracon))
    asrs['month'] = pd.Series(tracon_nar.index.map(get_month))
    asrs['year'] = pd.Series(tracon_nar.index.map(get_year))
    dicts = ['casa', 'faa', 'hand', 'iata_iaco', 'nasa']

    def num_months_between(month1, year1, month2, year2):
        return (year2 - year1) * 12 + month2 - month1

    """
    This returns a function to be applied row-wise on a dataframe. The inner function, when
    applied, returns a pandas series that finds which rows are within a month range. If
    the FAA/NTSB incident tracon_month occurs in January 2011, then the inner function finds
    all the rows that are within num_months months before January 2011.
    @param: month1 (int) 1-12
    @param: year1 (int)
    @param: num_months (int)
    @returns: function(row) that returns a pandas Series selecting the correct rows
    """
    def generate_compare(month1, year1, num_months = 1): # accident date
        def inner_func(row):
            month2, year2 = row['month'], row['year']
            n_m = num_months_between(month1, year1, month2, year2)
            return n_m > 0 and n_m <= num_months
        return inner_func
                    
    asrs_orig = asrs.merge(liwc_df.drop(['tracon', 'month', 'year'], axis = 1), on = 'tracon_month')

    # num_months = [1, 3, 6, 12, np.inf]
    num_months = [1, 3]
    for month_idx, n_month in enumerate(num_months):
        month_range_str = f'{n_month}m'
        if num_months == np.inf:
            month_range_str = 'atime'
        asrs = asrs_orig.copy()

        d2v_tm = pd.read_csv(f'./asrs_analysis/results/d2v_tracon_month_{col}_{n_month}mon.csv', index_col = 0)
        d2v_tm.index = d2v_tm.index.rename('tracon_month')
        d2v_tm.reset_index(inplace = True) 

        # combine with doc2vec 
        asrs = asrs.merge(d2v_tm, on = 'tracon_month', how = 'outer')
        # asrs = asrs.iloc[:1000]

        # this creates a dictionary from year/month -> pd.DataFrame of all the rows in 
        # the ASRS dataset within the month range (utilizing n_month)
        # ex.: January 2011, w/ n_month = 1 -> pd.DataFrame of all rows in ASRS in December 2010
        tracon_month_dict = {}
        month_year_df = airport_month_events[['month', 'year']].drop_duplicates()
        for idx, date_row in tqdm(month_year_df.iterrows(), total = month_year_df.shape[0], desc = \
                f"Creating year/month dictionary {n_month}mon"):
            code = ' '.join([str(date_row['month']), str(date_row['year'])])
            compare_func = generate_compare(date_row['month'], date_row['year'], num_months = n_month)
            tracon_month_dict[code] = asrs.loc[asrs.apply(compare_func, axis = 1), :].copy()

        # combines ASRS with incident/accident dataset (note d2v + liwc have already been merged to
        # ASRS). This utilizes the dictionary created above
        final_rows = []
        asrs_covered_ind = set()
        for idx, row in tqdm(airport_month_events.iterrows(), total = airport_month_events.shape[0], desc = \
        # for idx, row in tqdm(airport_month_events.iloc[:1000].iterrows(), total = 1000, desc = \
                f"Combining ASRS {n_month}mon"):
            code = ' '.join([str(row['month']), str(row['year'])])
            if code in tracon_month_dict:
                searched = tracon_month_dict[code]
                searched = searched.loc[searched['tracon'] == row['airport_code'], :]
                asrs_covered_ind.update(searched.index)

                tr_yr_mon = row[['airport_code', 'year', 'month']]

                if searched.shape[0] > 0:
                    cumulative = searched.drop(['tracon_month', 'tracon', 'year', 'month'], axis = 1).sum()
                    if month_idx == 0:
                        final_rows.append(pd.concat([row, cumulative], axis = 0))
                    else:
                        final_rows.append(pd.concat([tr_yr_mon, cumulative], axis = 0))
                else:
                    if month_idx == 0:
                        final_rows.append(pd.concat([row, pd.Series(index = asrs.columns.drop(\
                                ['tracon_month', 'tracon', 'year', 'month']), \
                                dtype = 'float64')], axis = 0))
                    else:
                        final_rows.append(pd.concat([tr_yr_mon, pd.Series(index = asrs.columns.drop(\
                                ['tracon_month', 'tracon', 'year', 'month']), \
                                dtype = 'float64')], axis = 0))

        cols = final_rows[0].index
        if missing_tracons is None:
            missing_tracons = []
            empty_row = pd.Series(index = cols, dtype = 'float64')

            total = unique_codes.shape[0] * num_time_periods
            for code_mon_yr in tqdm(product(unique_codes, range(1, 13), range(1988, 2020)), \
                    total = total, desc = "missing tracon_month"):
                if code_mon_yr not in all_combs:
                    code, month, year = code_mon_yr
                    e_r = empty_row.copy()
                    e_r['airport_code'] = code
                    e_r['year'] = year
                    e_r['month'] = month
                    missing_tracons.append(e_r)
        final_rows += missing_tracons

        print('% ASRS covered', len(asrs_covered_ind) / asrs.shape[0])
        print('% incident covered', len(asrs_covered_ind) / airport_month_events.shape[0])
        res = pd.DataFrame.from_dict({idx: row for idx, row in enumerate(final_rows)}, orient = 'index')

        # post-processing
        res = rename_cols(res, month_range_str, skip_cols = ame_cols)

        faa_ntsb_cols = ['ntsb_incidents', 'ntsb_accidents', 'faa_incidents']
        for fn_col in faa_ntsb_cols:
            if fn_col in res.columns:
                res.loc[res[fn_col].isna(), fn_col] = 0

        res = res.loc[:,~res.columns.duplicated()]
        res.set_index(['tracon_key', 'year', 'month'], inplace = True)
        print(res.shape)
        # res = reorder_cols(res)
        all_res.append(res)
        # res.to_csv(f'results/final_dataset_{col}_{n_month}mon.csv')
all_res = pd.concat(all_res, ignore_index = False, axis = 1)
all_res = reorder_cols(all_res)
all_res.to_csv('results/final_dataset.csv')
coverage = all_res.isna().sum()
coverage['total rows'] = all_res.shape[0]
coverage.to_csv('results/final_coverage.csv')
