import pandas as pd, numpy as np
ntsb = pd.read_csv('datasets/NTSB_AIDS_full.txt', sep = "|")
airports = pd.read_csv('datasets/airports.csv')

event_date = np.array([ntsb.loc[i, ' Event Date '].strip().split('/') for i in  np.arange(len(ntsb))]).T
month = event_date[0]
day = event_date[1]
year = event_date[2]

ntsb['year'] = year
ntsb['month'] = month

ntsb['airportcode_new'] = ntsb.loc[:, ' Airport Code '].copy().apply(str)
ntsb['airportname_new'] = ntsb.loc[:, ' Airport Name '].copy().apply(str)

ntsb = pd.get_dummies(ntsb, columns=[' Investigation Type '], drop_first=True)

ntsb_control = ntsb[['airportcode_new', 'airportname_new', 'year', 'month',\
        ' Investigation Type _ Accident ', ' Investigation Type _ Incident ', 'Event Id ']]\
        .groupby(by=['airportcode_new', 'airportname_new', 'year', 'month']).sum()

ntsb_control = ntsb_control.reset_index()

def isinvalid(entry): 
    if pd.isna(entry):
        return True
    if type(entry) == str: 
        if entry.strip() == "": 
            return True
        if entry.strip() == "N/A":
            return True
    return False

code_sel = ntsb_control['airportcode_new'].apply(isinvalid)
name_sel = ntsb_control['airportname_new'].apply(isinvalid)

# number of valid codes and names
print(ntsb_control.loc[~(code_sel & name_sel), :].shape)
print(ntsb_control.shape)

name_sel = ntsb[' Airport Name '].apply(isinvalid)
code_sel = ntsb[' Airport Code '].apply(isinvalid)
print(ntsb[~(name_sel & code_sel)].shape)

all_codes = set(airports['ident']) 
all_codes.update(airports['gps_code'])
all_codes.update(airports['local_code'])

print(np.sum(ntsb.loc[~(name_sel & code_sel), " Airport Code "].str.strip().apply(lambda x: x in all_codes)))

print(ntsb.loc[~(name_sel & code_sel)].shape)

ntsb_sub = ntsb.loc[~(name_sel & code_sel), :] 
sel = ntsb.loc[~(name_sel & code_sel), " Airport Code "].str.strip().apply(lambda x: x in all_codes)
filtered = ntsb_sub.loc[sel, :]

filtered = filtered.reset_index().drop('index', axis = 1)
event_date = np.array([filtered.loc[i, ' Event Date '].strip().split('/') for i in  np.arange(len(filtered))]).T
month = event_date[0]
day = event_date[1]
year = event_date[2]

filtered['year'] = year
filtered['month'] = month
filtered['airportcode_new'] = filtered.loc[:, ' Airport Code '].copy().apply(str)
filtered['airportname_new'] = filtered.loc[:, ' Airport Name '].copy().apply(str)
filtered['airportcode_new'] = filtered.loc[:, ' Airport Code '].str.strip()
filtered['airportname_new'] = filtered.loc[:, ' Airport Name '].str.strip()

output_df = filtered[['airportcode_new', 'airportname_new', 'year', 'month', \
        ' Investigation Type _ Accident ', ' Investigation Type _ Incident ', 'Event Id ']] \
        .groupby(by=['airportcode_new', 'airportname_new', 'year', 'month']).sum()

output_df.to_csv('results/NTSB_AIDS_full_output_new_jimmy.csv')
