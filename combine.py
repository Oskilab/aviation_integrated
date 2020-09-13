import pandas as pd, numpy as np
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

all_res = []
# for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
for col in ['narrative']:
    print('col', col)
    # asrs
    tracon_nar = pd.read_csv(f'asrs_analysis/results/tracon_month_{col}.csv', index_col = 0)

    # ntsb/faa incident dataset + volume
    airport_month_events = pd.read_csv('results/combined_vol_incident.csv', index_col = 0)
    ame_cols = list(airport_month_events.columns)

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

        # this creates a dictionary from year/month -> pd.DataFrame of all the rows in 
        # the ASRS dataset within the month range (utilizing n_month)
        # ex.: January 2011, w/ n_month = 1 -> pd.DataFrame of all rows in ASRS in December 2010
        tracon_month_dict = {}
        month_year_df = airport_month_events.iloc[:1000,:][['month', 'year']].drop_duplicates()
        for idx, date_row in tqdm(month_year_df.iterrows(), total = month_year_df.shape[0], desc = \
                f"Creating year/month dictionary {n_month}mon"):
            code = ' '.join([str(date_row['month']), str(date_row['year'])])
            compare_func = generate_compare(date_row['month'], date_row['year'], num_months = n_month)
            tracon_month_dict[code] = asrs.loc[asrs.apply(compare_func, axis = 1), :].copy()

        # combines ASRS with incident/accident dataset (note d2v + liwc have already been merged to
        # ASRS). This utilizes the dictionary created above
        final_rows = []
        asrs_covered_ind = set()
        # for idx, row in tqdm(airport_month_events.iloc[:1000].iterrows(), total = airport_month_events.shape[0], desc = \
        for idx, row in tqdm(airport_month_events.iloc[:1000].iterrows(), total = 1000, desc = \
                f"Combining ASRS {n_month}mon"):
            code = ' '.join([str(row['month']), str(row['year'])])
            if code in tracon_month_dict:
                searched = tracon_month_dict[code]
                searched = searched.loc[searched['tracon'] == row['airport_code'], :]
                asrs_covered_ind.update(searched.index)

                if searched.shape[0] > 0:
                    cumulative = searched.drop(['tracon_month', 'tracon', 'year', 'month'], axis = 1).sum()
                    if month_idx == 0:
                        final_rows.append(pd.concat([row, cumulative], axis = 0))
                    else:
                        final_rows.append(cumulative)
                else:
                    if month_idx == 0:
                        final_rows.append(pd.concat([row, pd.Series(index = asrs.columns.drop(\
                                ['tracon_month', 'tracon', 'year', 'month']))], axis = 0))
                    else:
                        final_rows.append(pd.Series(index = asrs.columns.drop(\
                                ['tracon_month', 'tracon', 'year', 'month'])), axis = 0)

        cols = final_rows[0].index
        empty_row = pd.Series(index = cols)

        unique_codes = airport_month_events['airport_code'].unique()
        all_combs = set()
        for idx, row in airport_month_events[['airport_code', 'year', 'month']].drop_duplicates() \
                .iterrows():
            code, mon, year = row['airport_code'], row['month'], row['year']
            all_combs.add((code, mon, year))

        total = unique_codes.shape[0] * num_time_periods
        for code_mon_yr in tqdm(product(unique_codes, range(1, 13), range(1988, 2020)), \
                total = total, desc = "missing tracon_month"):
            if code_mon_yr not in all_combs:
                code, month, year = code_mon_yr
                e_r = empty_row.copy()
                e_r['tracon_key'] = code
                e_r['year'] = year
                e_r['month'] = month
                final_rows.append(e_r)

        print('% ASRS covered', len(asrs_covered_ind) / asrs.shape[0])
        print('% incident covered', len(asrs_covered_ind) / airport_month_events.shape[0])
        res = pd.DataFrame.from_dict({idx: row for idx, row in enumerate(final_rows)}, orient = 'index')
        res = rename_cols(res, month_range_str, skip_cols = ame_cols)

        embed()
        res = res.loc[:,~res.columns.duplicated()]
        res.set_index(['tracon_key', 'year', 'month'], inplace = True)
        all_res.append(res)
        # res.to_csv(f'results/final_dataset_{col}_{n_month}mon.csv')
all_res = pd.concat(all_res, ignore_index = False, axis = 1)
embed()
all_res.to_csv('results/final_dataset.csv')
coverage = all_res.isna().sum()
coverage['total rows'] = all_res.shape[0]
coverage.to_csv('results/final_coverage.csv')
