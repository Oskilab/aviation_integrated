import pandas as pd, numpy as np
from IPython import embed
from tqdm import tqdm
"""
Combines ASRS data with FAA/NTSB, LIWC and Doc2Vec datasets
"""

# for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
for col in ['narrative']:
    print('col', col)
    # asrs
    tracon_nar = pd.read_csv(f'asrs_analysis/results/tracon_month_{col}.csv', index_col = 0)

    # ntsb/faa incident dataset + volume
    # airport_month_events = pd.read_csv('airport_month_events.csv', index_col = 0)
    airport_month_events = pd.read_csv('results/combined_vol_incident.csv', index_col = 0)

    # liwc counts
    liwc_df = pd.read_csv(f'asrs_analysis/results/liwc_tracon_month_{col}_counts.csv',index_col = 0, header = [1])
    # liwc_df = liwc_df.reset_index().rename({'index':'tracon_month'}, axis = 1)

    def get_tracon(x):
        split_x = str(x).split()
        if len(split_x) > 2:
            return " ".join(split_x[:-1])
        else:
            return split_x[0]

    def get_year(x):
        return int(str(x).split()[-1].split("/")[0])

    def get_month(x):
        return int(str(x).split()[-1].split("/")[1])


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

    num_months = [1, 3, 6, 12, np.inf]
    for n_month in num_months:
        asrs = asrs_orig.copy()

        d2v_tm = pd.read_csv(f'./asrs_analysis/results/d2v_tracon_month_{col}_{n_month}mon.csv', index_col = 0)
        d2v_tm.index = d2v_tm.index.rename('tracon_month')
        d2v_tm.reset_index(inplace = True) 

        # combine with doc2vec 
        asrs = asrs.merge(d2v_tm, on = 'tracon_month')

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
                f"Combining ASRS {n_month}mon"):
            code = ' '.join([str(row['month']), str(row['year'])])
            if code in tracon_month_dict:
                searched = tracon_month_dict[code]
                searched = searched.loc[searched['tracon'] == row['airport_code'], :]
                asrs_covered_ind.update(searched.index)

                if searched.shape[0] > 0:
                    cumulative = searched.drop(['tracon_month', 'tracon', 'year', 'month'], axis = 1).sum()
                    final_rows.append(pd.concat([row, cumulative], axis = 0))
                else:
                    final_rows.append(pd.concat([row, pd.Series(index = asrs.columns.drop(\
                            ['tracon_month', 'tracon', 'year', 'month']))], axis = 0))

        print('% ASRS covered', len(asrs_covered_ind) / asrs.shape[0])
        print('% incident covered', len(asrs_covered_ind) / airport_month_events.shape[0])
        res = pd.DataFrame.from_dict({idx: row for idx, row in enumerate(final_rows)}, orient = 'index')
        res.to_csv(f'results/final_dataset_{col}_{n_month}mon.csv')
