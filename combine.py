import pandas as pd, numpy as np
from IPython import embed
from tqdm import tqdm
"""
Combines ASRS data with FAA/NTSB, LIWC
"""

# need to iterate over col = narrative, synopsis, combined
for col in ['narrative', 'synopsis', 'combined']:
    print('col', col)
    # asrs
    tracon_nar = pd.read_csv('abrev_datasets/tracon_month_{col}.csv', index_col = 0)

    # ntsb/faa incident dataset + volume
    # airport_month_events = pd.read_csv('airport_month_events.csv', index_col = 0)
    airport_month_events = pd.read_csv('results/combined_vol_incident.csv', index_col = 0)

    # abreviation counts
    # total_cts = pd.read_csv('abrev_datasets/total_cts_tagged_{col}.csv')

    # liwc counts
    liwc_df = pd.read_csv('abrev_datasets/liwc_tracon_month_{col}_counts.csv',index_col = 0, header = [1])
    # liwc_df = liwc_df.reset_index().rename({'index':'tracon_month'}, axis = 1)

    def get_tracon(x):
        return str(x).split()[0]

    def get_year(x):
        return int(str(x).split()[1].split("/")[0])

    def get_month(x):
        return int(str(x).split()[1].split("/")[1])

    tracon = pd.Series(liwc_df.index.map(get_tracon), index = liwc_df.index)
    year = pd.Series(liwc_df.index.map(get_year), index = liwc_df.index)
    month = pd.Series(liwc_df.index.map(get_month), index = liwc_df.index)

    liwc_df['tracon'] = tracon
    liwc_df['month'] = month
    liwc_df['year'] = year

    liwc_df = liwc_df.reset_index().rename({'index':'tracon_month'}, axis = 1)

    tracon = pd.Series(tracon_nar.index.map(get_tracon))
    year = pd.Series(tracon_nar.index.map(get_year))
    month = pd.Series(tracon_nar.index.map(get_month))

    asrs = tracon_nar.reset_index()
    asrs['tracon'] = tracon
    asrs['month'] = month
    asrs['year'] = year
    dicts = ['casa', 'faa', 'hand', 'iata_iaco', 'nasa']

    def num_months_between(month1, year1, month2, year2):
        return (year2 - year1) * 12 + month2 - month1

    def generate_compare(month1, year1, num_months = 1): # accident date
        def inner_func(row):
            month2, year2 = row['month'], row['year']
            n_m = num_months_between(month1, year1, month2, year2)
            return n_m > 0 and n_m <= num_months
        return inner_func
                    
    asrs_orig = asrs.merge(liwc_df.drop(['tracon', 'month', 'year'], axis = 1), on = 'tracon_month')

    num_months = [1, 3, 6, 12, np.inf]
    for n_month in num_months:
        print('month window', n_month)
        asrs = asrs_orig.copy()

        d2v_tm = pd.read_csv(f'./abrev_datasets/d2v_tracon_month_{col}_{n_month}mon.csv', index_col = 0)
        d2v_tm.index = d2v_tm.index.rename('tracon_month')
        d2v_tm.reset_index(inplace = True) 

        asrs = asrs.merge(d2v_tm, on = 'tracon_month')

        tracon_month_dict = {}
        print('first')
        for idx, date_row in tqdm(airport_month_events[['month', 'year']].drop_duplicates().iterrows()):
            code = ' '.join([str(date_row['month']), str(date_row['year'])])
            compare_func = generate_compare(date_row['month'], date_row['year'], num_months = n_month)
            tracon_month_dict[code] = asrs.loc[asrs.apply(compare_func, axis = 1), :].copy()

        final_rows = []
        print('second')
        asrs_covered_ind = set()
        for idx, row in tqdm(airport_month_events.iterrows()):
            code = ' '.join([str(row['month']), str(row['year'])])
            if code in tracon_month_dict:
                searched = tracon_month_dict[code]
                searched = searched.loc[searched['tracon'] == row['airport_code'], :]
                asrs_covered_ind.update(searched.index)

                if searched.shape[0] > 0:
                    cumulative = searched.drop(['tracon_month', 'tracon'], axis = 1).sum()
                    final_rows.append(pd.concat([row, cumulative], axis = 0))
                else:
                    final_rows.append(pd.concat([row, pd.Series(index = asrs.columns)], axis = 0))

        print('% covered', len(asrs_covered_ind) / asrs.shape[0])
        res = pd.DataFrame.from_dict({idx: row for idx, row in enumerate(final_rows)}, orient = 'index')
        res.to_csv(f'results/final_dataset_{n_month}mon.csv')
