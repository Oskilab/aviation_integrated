import pandas as pd, numpy as np, re, os
from tqdm import tqdm
from IPython import embed
from collections import namedtuple
coverage = namedtuple('coverage', ['part', 'total'])
"""
Combines flight volume data (WEB-Report-*) with airport_month_events, which is the processed
FAA/NTSB incident/accident data
"""

tracon_fns = ['51547', '49612', '29844', '52212', '81669', '22012', '59543']
tracon_fns = set([f'WEB-Report-{x}.csv' for x in tracon_fns])

def get_first_header_idx(df):
    for idx in range(df.shape[0]):
        if 'IFR Itinerant' in set(df.iloc[idx, :]):
            return idx

def generate_prefix_header(header_row):
    prefixes = []
    curr_prefix = ''
    for x in header_row:
        if x == ' ':
            prefixes.append('')
        elif not pd.isna(x):
            curr_prefix = x + '\t'
            prefixes.append(curr_prefix)
        else:
            prefixes.append(curr_prefix)
    return prefixes

def generate_column_names(col_row, prefix):
    final_cols = []
    enter = '\r\n'
    for idx, elem in enumerate(col_row):
        if pd.isna(elem):
            final_cols.append('')
        else:
            final_cols.append(f"{prefix[idx]}{elem.replace(enter, ' ')}")
    return final_cols

def load_volume_file(filename):
    global tracon_fns

    vol_data = pd.read_csv(filename)
    first_header_idx = get_first_header_idx(vol_data)
    
    # because the csv has multiple headers, we create a list of column name prefixes
    prefixes = generate_prefix_header(vol_data.iloc[first_header_idx, :])
            
    # generate column names
    cols = generate_column_names(vol_data.iloc[first_header_idx + 1, :], prefixes)

    vol_data.columns = cols
    vol_data = vol_data.iloc[3:-2,:-1].copy() # some beginning rows and end rows are dropped

    vol_data = vol_data.loc[:, ~vol_data.columns.duplicated()].copy()
    return vol_data

def load_volume():
    # read in flight volume data
    pds = []
    for x in os.listdir('datasets/'):
        if '.csv' in x and '.swp' not in x and 'WEB-Report' in x:
            vol_data = load_volume_file(f"datasets/{x}")

            # add column indicating from tracon or tower
            if x in tracon_fns:
                vol_data['ATADS_Tracon_Tower'] = 'Tracon'
            else:
                vol_data['ATADS_Tracon_Tower'] = 'Tower'

            pds.append(vol_data)

    return pd.concat(pds, axis = 0, ignore_index = True, sort = True)

def conv_to_int(x):
    try:
        return int(x)
    except:
        return np.nan

def process_dates(vol_data):
    vol_data['month'] = vol_data['Date'].str.split("/")\
            .apply(lambda x: conv_to_int(x[0]))
    vol_data['year'] = vol_data['Date'].str.split("/")\
            .apply(lambda x: conv_to_int(x[1]) if len(x) > 1 else np.nan)
    vol_data = vol_data[~(vol_data['month'].isna() | vol_data['year'].isna())].copy()
    vol_data['month'] = vol_data['month'].astype(int)
    vol_data['year'] = vol_data['year'].astype(int)
    return vol_data
    
def generate_facility_date_dict(vol_data):
    # create dictionary of facility/date -> loc idx (it's a nested dictionary using facility and date
    # consecutively)
    id_to_idx = {}
    for idx, row in tqdm(vol_data.iterrows(), total = vol_data.shape[0]):
        code = str(row['Facility']).strip()
        facility_dict = id_to_idx.get(code, {})
        facility_dict[row['Date']] = idx
        id_to_idx[code] = facility_dict
    return id_to_idx

def date_to_str(year, month):
    month_str = '0' * int(month < 10) + str(month)
    year_str = str(year)
    return f"{month_str}/{year_str}"

def generate_final_row(row, code, date_str, id_to_idx, vol_data):
    if code in id_to_idx and date_str in id_to_idx[code]:
        final_row = pd.concat([row, vol_data.loc[id_to_idx[code][date_str], :]], axis = 0)
    else:
        final_row = pd.concat([row, pd.Series(index=vol_data.columns)], axis = 0)
    return final_row[~final_row.index.duplicated(keep='first')]

def generate_combined_dfs(res, vol_data, id_to_idx):
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
        final_row = generate_final_row(row, code, date_str, id_to_idx, vol_data)

        # housekeeping
        if code in id_to_idx and date_str not in id_to_idx:
            date_nf_ct += 1
            rows_dnf.append(final_row)
        elif code not in id_to_idx:
            code_nf_ct += 1
            rows_cnf.append(final_row)

        rows.append(final_row)

    cnf = pd.DataFrame(rows_cnf)
    dnf = pd.DataFrame(rows_dnf)
    all_rows = pd.DataFrame(rows)
    return all_rows, cnf, dnf

def main():
    vol_data = load_volume()
    vol_data = process_dates(vol_data)

    id_to_idx = generate_facility_date_dict(vol_data)

    # combine the two csvs
    res = pd.read_csv('results/airport_month_events.csv', index_col = 0)

    # remove this later
    top_50_iata = set(pd.read_excel('datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    res = res[res['airport_code'].apply(lambda x: x in top_50_iata)]

    all_rows, cnf, dnf = generate_combined_dfs(res, vol_data, id_to_idx)

    # save csvs
    all_rows.to_csv('results/combined_vol_incident.csv', index = False)
    cnf.to_csv('results/nf_codes.csv')
    dnf.to_csv('results/nf_dates.csv')

if __name__ == "__main__":
    main()
