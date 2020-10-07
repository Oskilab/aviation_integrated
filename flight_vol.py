import pandas as pd, numpy as np, re, os
from tqdm import tqdm
from IPython import embed
from collections import namedtuple
coverage = namedtuple('coverage', ['part', 'total'])
"""
Combines flight volume data (WEB-Report-*) with airport_month_events, which is the processed
FAA/NTSB incident/accident data
"""

tracon_fns = ['51547', '49612', '29844', '52212', '81669', '22012']
tracon_fns = set([f'WEB-Report-{x}.csv' for x in tracon_fns])

# read in flight volume data
pds = []
for x in os.listdir('datasets/'):
    if '.csv' in x and '.swp' not in x and 'WEB-Report' in x:
        print(x)
        part2 = pd.read_csv(f"datasets/{x}")
        
        # because the csv has multiple headers, we create a list of column name prefixes
        prefixes = []
        curr_prefix = ''
        for x in part2.iloc[1, :]:
            if x == ' ':
                prefixes.append('')
            elif not pd.isna(x):
                curr_prefix = x + '\t'
                prefixes.append(curr_prefix)
            else:
                prefixes.append(curr_prefix)
                
        # generate column names
        final_cols = []
        enter = '\r\n'
        for idx, elem in enumerate(part2.iloc[2, :]):
            if pd.isna(elem):
                final_cols.append('')
            else:
                final_cols.append(f"{prefixes[idx]}{elem.replace(enter, ' ')}")

        part2.columns = final_cols
        part2 = part2.iloc[3:-2,:-1].copy() # some beginning rows and end rows are dropped

        if x in tracon_fns:
            part2['ATADS_Tracon_Tower'] = 'Tracon'
        else:
            part2['ATADS_Tracon_Tower'] = 'Tower'
        pds.append(part2)
part2 = pd.concat(pds, axis = 0, ignore_index = True, sort = True)

part2['month'] = part2['Date'].str.split("/").apply(lambda x: int(x[0]) if isinstance(x, list) else np.nan)
part2['year'] = part2['Date'].str.split("/").apply(lambda x: int(x[1]) if isinstance(x, list) else np.nan)

# create dictionary of facility/date -> loc idx (it's a nested dictionary using facility and date
# consecutively)
id_to_idx = {}
for idx, row in tqdm(part2.iterrows(), total = part2.shape[0]):
    code = row['Facility'].strip()
    facility_dict = id_to_idx.get(code, {})
    facility_dict[row['Date']] = idx
    id_to_idx[code] = facility_dict

# combine the two csvs
res = pd.read_csv('results/airport_month_events.csv', index_col = 0)
num_cols = part2.columns.shape[0]
all_nans = pd.Series(index = part2.columns)

def date_to_str(year, month):
    month_str = '0' * int(month < 10) + str(month)
    year_str = str(year)
    return f"{month_str}/{year_str}"

# generate dataframe. If the incident/accident date is not found in he flight volume data, then
# it is added to rows_dnf (rows date not found), and if the tracon code is not found in the volume
# data it is added to rows_cnf (rows code not found). Note nans are used to replace the missing info
rows, rows_dnf, rows_cnf = [], [], []
date_nf_ct, code_nf_ct = 0, 0
missing_code = 0
for idx, row in tqdm(res.iterrows(), total = res.shape[0]):
    code = row['airport_code']
    if pd.isna(code):
        missing_code += 1
        continue
    code = code.strip()
    date_str = date_to_str(row['year'], row['month'])
    final_row = None
    
    if code in id_to_idx:
        if date_str in id_to_idx[code]:
            final_row = pd.concat([row, part2.loc[id_to_idx[code][date_str], :]], axis = 0).drop_duplicates()
        else:
            date_nf_ct += 1
            final_row = pd.concat([row, all_nans], axis = 0).drop_duplicates()
            rows_dnf.append(final_row)
    else:
        code_nf_ct += 1
        final_row = pd.concat([row, all_nans], axis = 0).drop_duplicates()
        rows_cnf.append(final_row)
    rows.append(final_row)

num_zzz = res.loc[(res['airport_code'] == 'ZZZ') | (res['airport_code'] == 'ZZZZ'), :].shape[0]
# find coverage of incident/accident data
print(coverage(res.shape[0] - missing_code - num_zzz, res.shape[0] - date_nf_ct - \
        code_nf_ct - missing_code - num_zzz), missing_code, num_zzz)

cnf = pd.DataFrame(rows_cnf)
dnf = pd.DataFrame(rows_dnf)
all_rows = pd.DataFrame(rows)

# save csvs
all_rows.to_csv('results/combined_vol_incident.csv', index = False)
cnf.to_csv('results/nf_codes.csv')
dnf.to_csv('results/nf_dates.csv')
