import pandas as pd, numpy as np, re, os
from tqdm import tqdm
from IPython import embed
from collections import namedtuple
from itertools import product

coverage = namedtuple('coverage', ['part', 'total'])
"""
Combines flight volume data (WEB-Report-*) with airport_month_events, which is the processed
FAA/NTSB incident/accident data
"""

tracon_fns = ['51547', '49612', '29844', '52212', '81669', '22012', '59543']
tracon_fns = set([f'WEB-Report-{x}.csv' for x in tracon_fns])

def get_first_header_idx(df):
    """
    Each volume xlsx file contains a number of extraneous rows at the top.
    This function calculates the first row which contains the header info
    @param: df (pd.DataFrame), volume xlsx file
    """
    for idx in range(df.shape[0]):
        if 'IFR Itinerant' in set(df.iloc[idx, :]):
            return idx

def generate_prefix_header(header_row):
    """
    Each volume xlsx file contains two rows that contain information about
    the columns. The first row is utilized as a prefix (we utilize a tab
    to separate the prefix and the column name)
    @param: header_row (pd.Series), the header row
    @returns: prefix (list of str), contains same number of elements as
        the header_row. Each string is the prefix for that particular column
    """
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
    """
    This generates the columns for a given volume xlsx file. It utilizes
    the prefix elements that were extracted and adds the column info.
    @param: col_row (pd.Series), the row that contains the name of the column
    @param: prefix (list of str), with same number of elements as col_row
        each str is the prefix for that particular column
    @returns: final_cols (list of str), with same number of elements as col_row
        contains final column names
    """
    final_cols = []
    enter = '\r\n'
    for idx, elem in enumerate(col_row):
        if pd.isna(elem):
            final_cols.append('')
        else:
            final_cols.append(f"{prefix[idx]}{elem.replace(enter, ' ')}")
    return final_cols

def load_volume_file(filename):
    """
    This loads a volume xlsx file, generates column names, removes extraneous
    info, and removes duplicated columns
    @param: filename (str) path to volume file
    @returns: vol_data (pd.DataFrame) of volume data
    """
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
    """
    This loads in all volume xlsx files and concatenates them together. Must be
    in datasets folder with WEB-Report in the title.
    """
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
    """
    Helper function to convert string to int, nan when error occurs.
    @param: x (string) to be converted
    @returns: int/np.nan of converted x
    """
    try:
        return int(x)
    except:
        return np.nan

def process_dates(vol_data):
    """
    Processes all the dates in the volume data, and removes nans.
    @param: vol_data (pd.DataFrame) the full volume dataset
    @returns: vol_data (pd.DataFrame) full volume dataset with cleaned dates
    """
    vol_data['month'] = vol_data['Date'].str.split("/")\
            .apply(lambda x: conv_to_int(x[0]))
    vol_data['year'] = vol_data['Date'].str.split("/")\
            .apply(lambda x: conv_to_int(x[1]) if len(x) > 1 else np.nan)
    vol_data = vol_data[~(vol_data['month'].isna() | vol_data['year'].isna())].copy()
    vol_data['month'] = vol_data['month'].astype(int)
    vol_data['year'] = vol_data['year'].astype(int)
    return vol_data
    
def generate_facility_date_dict(vol_data):
    """
    This creates a nested dictionary dict[facility] -> dict[date] -> location
    @param: vol_data (pd.DataFrame) the full volume dataset.
    @returns: dict[facility] -> dict[date] -> location in vol_data
    """
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
    """
    Converts year/month pair to a string of month/year with padded month
    @param: year (int/np.nan) 
    @param: month (int/np.nan) 
    @returns month/year
    """
    month_str = '0' * int(month < 10) + str(month)
    year_str = str(year)
    return f"{month_str}/{year_str}"

def generate_final_row(row, code, date_str, id_to_idx, vol_data):
    """
    This combines a row from the incident/accident dataset with the volume data.
    @param: row (pd.Series), row from incident/accident dataset (airport_month_events.csv)
    @param: date_str (str) comptued from date_to_str function
    @param: id_to_idx (nested dict), computed from generate_facility_date_dict function
    @param: vol_data (pd.DataFrame) the full volume dataset.
    @returns: final_row (pd.Series), the combined row from incident/accident dataset
        and the volume dataset. Nans if the volume data could not be found
    """
    if code in id_to_idx and date_str in id_to_idx[code]:
        final_row = pd.concat([row, vol_data.loc[id_to_idx[code][date_str], :]], axis = 0)
    else:
        final_row = pd.concat([row, pd.Series(index=vol_data.columns)], axis = 0)
    return final_row[~final_row.index.duplicated(keep='first')]

def generate_combined_dfs(res, vol_data, id_to_idx):
    """
    This generates the combined dataset of the volume data and the incident/accident dataset.
    @param: res (pd.DataFrame) the incident/accident dataset
    @param: vol_data (pd.DataFrame) the volume data
    @param: id_to_idx (nested dict), computed from generate_Facility_date_dict function
    @returns: all_rows (pd.DataFrame) full combined dataset
    @returns: cnf (pd.DataFrame) all rows where the code was not found in volume data
    @returns: dnf (pd.DataFrame) all rows where the code was found but the date was not found
        in the volume data
    """
    # generate dataframe. If the incident/accident date is not found in he flight volume data, then
    # it is added to rows_dnf (rows date not found), and if the tracon code is not found in the volume
    # data it is added to rows_cnf (rows code not found). Note nans are used to replace the missing info
    rows, rows_dnf, rows_cnf = [], [], []
    date_nf_ct, code_nf_ct = 0, 0
    for idx, row in tqdm(res.iterrows(), total = res.shape[0]):
        code = row['airport_code']
        if pd.isna(code):
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

    # top 50 iata codes
    top_50_iata = set(pd.read_excel('datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    res = res[res['airport_code'].apply(lambda x: x in top_50_iata)]

    all_rows, cnf, dnf = generate_combined_dfs(res, vol_data, id_to_idx)

    # save csvs
    all_rows.to_csv('results/combined_vol_incident.csv', index = False)
    cnf.to_csv('results/nf_codes.csv')
    dnf.to_csv('results/nf_dates.csv')

if __name__ == "__main__":
    main()
