import pandas as pd, numpy as np, re
from tqdm import tqdm
pds = []
for x in os.listdir('datasets/'):
    if '.csv' in x and '.swp' not in x and 'WEB-Report' in x:
        print(x)
        part2 = pd.read_csv(f"datasets/{x}")
        
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
                
        final_cols = []
        enter = '\r\n'
        for idx, elem in enumerate(part2.iloc[2, :]):
            if pd.isna(elem):
                final_cols.append('')
            else:
                final_cols.append(f"{prefixes[idx]}{elem.replace(enter, ' ')}")
        part2.columns = final_cols
        part2 = part2.iloc[3:-2,:-1].copy()
        pds.append(part2)
part2 = pd.concat(pds, axis = 0, ignore_index = True)

def check_format(df):
    pat = re.compile('From (\d{2}/\d{4}) To (\d{2}/\d{4})')
    mat = pat.match(df.iloc[0, 0])
    assert(mat is not None)
    print(mat.group(1), mat.group(2))

check_format(part2)

part2['month'] = part2['Date'].str.split("/").apply(lambda x: int(x[0]) if isinstance(x, list) else np.nan)
part2['year'] = part2['Date'].str.split("/").apply(lambda x: int(x[1]) if isinstance(x, list) else np.nan)

# create dictionary of facility/date -> loc idx
id_to_idx = {}
for idx, row in part2.iterrows():
    code = row['Facility'].strip()
    facility_dict = id_to_idx.get(code, {})
    facility_dict[row['Date']] = idx
    id_to_idx[code] = facility_dict

# combine the two csvs
res = pd.read_csv('airport_month_events.csv', index_col = 0)
num_cols = part2.columns.shape[0]
all_nans = pd.Series(index = part2.columns)

def date_to_str(year, month):
    month_str = '0' * int(month < 10) + str(month)
    year_str = str(year)
    return f"{month_str}/{year_str}"

rows, rows_dnf, rows_cnf = [], [], []
date_nf_ct, code_nf_ct = 0, 0
for idx, row in tqdm(res.iterrows()):
    code = row['airport_code'].strip()
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

cnf = pd.DataFrame(rows_cnf)
dnf = pd.DataFrame(rows_dnf)
all_rows = pd.DataFrame(rows)

all_rows.to_csv('combined_vol_incident.csv')
cnf.to_csv('nf_codes.csv')
dnf.to_csv('nf_dates.csv')
