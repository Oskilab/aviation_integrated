from collections import Counter
from IPython import embed
import pandas as pd, numpy as np, matplotlib.pyplot as plt
import re, os

def separate_atc_info(row):
    """
    This is applied onto a dataframe to separate firstpass into atc data and info data.
        [(atc_code0, atc_type0, atc_repeated0), (info_code0, info_type0, info_repeated0)...]
        -> [[(atc_code0, ...), (atc_code1...),...], [(info_code0, ...), (info_code1...),...]
    @param: row (pd.Series) to be separated
    @returns: [atc_list, info_list] (list of lists) first representing atc code
        second represents miscellaneous information
    """
    global all_codes # to avoid recreating same set repeatedly
    atc_list, info_list = [], []
    for elem in row.firstpass:
        if elem[1] in all_codes:
            atc_list.append(elem)
        else:
            info_list.append(elem)
    return [atc_list, info_list]

def separate_nonatc(df):
    """
    Creates two columns (atc_list, info_list) consisting of atc data and info data from
    the original column with all the data.
    @param: df (pd.DataFrame) with firstpass as a column
    @returns: new df with added columns
    """
    meta_list = df.apply(separate_atc_info, axis = 1)
    df['atc_list'] = meta_list.apply(lambda x: x[0])
    df['info_list'] = meta_list.apply(lambda x: x[1])
    return df

def apply_getcode(ind, elem):
    """
    Returns the atc/info code
    @param: ind (int) that indexes into elem. If ind = 0, for example, this function
        will return the 1st code listed in elem
    @param: elem (list of tuples) [(atc_type0, atc_code0, atc_repeated0)...]
    @returns: atc_code[ind]
    """
    if len(elem) > ind:
        return elem[ind][1]
    else:
        return ""
def apply_gettype(ind, elem):
    """
    Returns the atc/info type
    @param: ind (int) that indexes into elem. If ind = 0, for example, this function
        will return the 1st type listed in elem
    @param: elem (list of tuples) [(atc_type0, atc_code0, atc_repeated0)...]
    @returns: atc_type[ind]
    """
    if len(elem) > ind:
        return elem[ind][0]
    else:
        return ""
def apply_getrep(ind, elem):
    """
    Returns the atc/info repeated
    @param: ind (int) that indexes into elem. If ind = 0, for example, this function
        will return the 1st code listed in elem
    @param: elem (list of tuples) [(atc_type0, atc_code0, atc_repeated0)...]
    @returns: atc_repeated[ind]
    """
    if len(elem) > ind:
        return not re.match(elem[ind][2], r'^ *$')
    else:
        return False

def expand_atc(df):
    """
    Given a dataframe with atc_list and info_list, this function adds columns for
    (atc|info)_(code|type|repeated)(num). 
        atc_list = (atc_type0, atc_code0, atc_repeated0), (atc_code1...), ...
        atc_code0 = atc_code0, atc_code1, ...
        atc_type0 = atc_type0, ...
        ...
    @param: df (pd.DataFrame) with atc_list and info_list
    """
    max_atcs = df['atc_list'].apply(len).max()
    for i in range(max_atcs):
        df[f'atc_code{i}'] =  df['atc_list'].apply(lambda x: apply_getcode(i, x))
        df[f'atc_type{i}'] = df['atc_list'].apply(lambda x: apply_gettype(i, x))
        df[f'atc_repeated{i}'] = df['atc_list'].apply(lambda x: apply_getrep(i, x))
    max_info = df['info_list'].apply(len).max()
    for i in range(max_info):
        df[f'info_code{i}'] =  df['info_list'].apply(lambda x: apply_getcode(i, x))
        df[f'info_type{i}'] = df['info_list'].apply(lambda x: apply_gettype(i, x))
        df[f'info_repeated{i}'] = df['info_list'].apply(lambda x: apply_getrep(i, x))
    return df
    
# true data
datasets_folder = 'datasets/ASRS Data/'
all_acs, all_fns = [], list(os.listdir(datasets_folder))
for fn in all_fns:
    # one file is not structured like the others
    if ".swp" not in fn and fn != "8-2019 - 6-2020-ASRS_DBOnline-2.csv":
        all_acs.append(pd.read_csv(f'{datasets_folder}{fn}'))

ac = pd.concat(all_acs, axis = 0, ignore_index = True)

# create sets
wiki_pd = pd.read_csv('datasets/wiki_code.csv')
tracon_pd = pd.read_csv('datasets/tracon_code.csv')

all_codes = set(wiki_pd['IATA'])
all_codes.update(wiki_pd['ICAO'])
all_codes.update(tracon_pd['LocID'])
all_codes.remove(np.nan)

# preprocess atc_advisory fields
atc_adv_cols = ['atc_advisory_acft1', 'atc_advisory_acft2']
for atc_col in atc_adv_cols:
    # remove all np nans and turn them into empty strings
    ac.loc[ac[atc_col].isna(), atc_col] = ''
ac['atcadvisory'] = ac[atc_adv_cols[0]] + '; ' + ac[atc_adv_cols[1]]

# remove the "; " at the end of atcadvisory
ends_with_sel = ac['atcadvisory'].str.endswith('; ')
ac.loc[ends_with_sel, 'atcadvisory'] = ac.loc[ends_with_sel, 'atcadvisory'].str.slice(stop = -2)

# Find Codes and Split. This creates a pd.Series where each element is a list consisting of tuples
# in the form of (atc type, atc code, possible repeated atc type).
res = ac['atcadvisory'].str.findall(r'(Center|Tower|TRACON|FSS|Ground|Ramp|CTAF|UNICOM|Military Facility)'
        ' ([A-Za-z0-9]{2,10})(\.ARTCC|\.Tower|\.FSS|\.TRACON|\.MILFAC|\&\.Tower|\.UNICOM){0,1}')

subset = pd.DataFrame({'atcadvisory': ac['atcadvisory'], 'firstpass': res})

def remove_firstpass(row):
    """
    Takes firstpass (generated above) and removes the information found by the first pass from
    the original column (atcadvisory).
    @param: row (pd.Series) of a given dataframe. Needs columns atcadvisory and firstpass
    @returns: string without captured data
    """
    s = row['atcadvisory']
    s = s.replace(' ; ', ' ')
    s = s.replace('; ', ' ')
    s = s.replace(' ;', ' ')
    for elem in row['firstpass']:
        s = s.replace(' '.join(elem[:-1]) + elem[-1], '')
    return s

# our goal now is to make sure all the removed_firstpass becomes empty string
subset['removed_firstpass'] = subset.apply(remove_firstpass, axis = 1)
# embed() run "subset['removed_firstpass'].unique()"
# 1/0

# Deal with some leftover garbage (only TRACON/Tower left), this only finds 2 rows
d = set(['TRACON', 'Tower', ''])
all_idx = []
for row in subset['removed_firstpass'].iteritems():
    all_d = True
    for elem in row[1].split(' '):
        if elem not in d:
            all_d = False
    if all_d:
        all_idx.append(row[0])
subset.loc[all_idx, 'removed_firstpass'] = '' 
leftovers = subset[~subset['removed_firstpass'].str.match(r'^ *$')]
leftovers.to_csv('outputs/leftovers.csv')

# MANUAL FIXING. Check leftovers.csv if you need to double check
fixed_indices = [23223, 50769, 68226, 79476, 91462, 92231, 95644, \
        95925, 107140, 116490, 133368, 150243, 153256, 154491, 208448, \
        224205, 224497]
subset.loc[23223, 'firstpass'] = subset.loc[23223, 'firstpass'][:-1] + \
        [('Tower', 'ZZZ1', '')]
subset.loc[50769, 'firstpass'] = subset.loc[50769, 'firstpass'][:2] + \
        [('UNICOM', '148.30', '')] + subset.loc[50769, 'firstpass'][-1:]
subset.loc[68226, 'firstpass'] = subset.loc[68226, 'firstpass'][:1] + \
        [('UNICOM', 'FBO', '')] 
subset.loc[79476, 'firstpass'] = subset.loc[79476, 'firstpass'][:-1] + \
        [('Military Facility', 'RIDGE CTRL', '')]
subset.loc[91462, 'firstpass'] = subset.loc[91462, 'firstpass'][:1] + \
        [('Military Facility', 'ZBW CAPE', '')] + subset.loc[91462, 'firstpass'][2:]
subset.loc[92231, 'firstpass'] = subset.loc[92231, 'firstpass'][:-1] + \
        [('FSS', 'JNU', ''), ('FSS', 'RDU', '')] # Both of these are found, so separated
subset.loc[95644, 'firstpass'] = subset.loc[95644, 'firstpass'][:2] + \
        [('UNICOM', '998.00', ''), ('UNICOM', '159', '')]
subset.loc[95925, 'firstpass'] = subset.loc[95925, 'firstpass'][:2] + \
        [('UNICOM', '123.14', ''), ('UNICOM', '159', '')]
subset.loc[107140, 'firstpass'] = []
subset.loc[116490, 'firstpass'] = subset.loc[116490, 'firstpass'][:-1] + \
        [('Tower', 'ZZZ1', '')]
subset.loc[133368, 'firstpass'] = subset.loc[133368, 'firstpass'][:1] + \
        [('UNICOM', '0', '')] + subset.loc[133368, 'firstpass'][1:]
subset.loc[150243, 'firstpass'].append(('UNICOM', 'I23', ''))
subset.loc[153256, 'firstpass'] = [('UNICOM', '122.9', '')]
subset.loc[154491, 'firstpass'] = subset.loc[154491, 'firstpass'][:-1] + \
        [('TRACON', 'ZZZZ', '')]
subset.loc[208448, 'firstpass'].append(('TRACON', 'O', ''))
subset.loc[224205, 'firstpass'] = subset.loc[224205, 'firstpass'][:-1] + \
        [('UNICOM', '998.00', '')]
subset.loc[224497, 'firstpass'] = subset.loc[224497, 'firstpass'][:-1] + \
        [('UNICOM', '123.14', '')]
subset.loc[fixed_indices, 'remove_firstpass'] = ''

subset = separate_nonatc(subset)
subset = expand_atc(subset)

# Create dictionaries of the most common info codes and types
# key -> value: code -> number of times it is used
code_ct_dict, type_ct_dict = {}, {}
for i in range(6):
    cts = subset[f'info_code{i}'].value_counts()
    for idx, code in enumerate(cts.index):
        if code == '':
            continue
        elif code in code_ct_dict:
            code_ct_dict[code] += cts[idx]
        else:
            code_ct_dict[code] = cts[idx]
    cts = subset[f'info_type{i}'].value_counts()
    for idx, code in enumerate(cts.index):
        if code == '':
            continue
        elif code in type_ct_dict:
            type_ct_dict[code] += cts[idx]
        else:
            type_ct_dict[code] = cts[idx]

type_df = pd.DataFrame({'type': list(type_ct_dict.keys()), 'ct': list(type_ct_dict.values())})
type_df.sort_values(by = 'ct', ascending = False)

# creates barplot of most common words (types)
plt.title("Most Common Words (not ATC)")
type_df.iloc[:10].plot.barh(x = 'type', y = 'ct')
plt.xlabel("Count")
plt.ylabel("Words")
plt.savefig("outputs/type_cts.png")
type_df.to_csv("outputs/type_cts.csv", index = False)

code_df = pd.DataFrame({'codes': list(code_ct_dict.keys()), 'ct': list(code_ct_dict.values())})
code_df.sort_values(by = 'ct', ascending = False)
# creates barplot of most common words (codeS)
plt.title("Most Common Words (not ATC)")
code_df.iloc[:10].plot.barh(x = 'codes', y = 'ct')
plt.xlabel("Count")
plt.ylabel("Words")
plt.savefig("outputs/codes_cts.png")
code_df.to_csv("outputs/codes_cts.csv", index = False)

# add back to original_dataframe
pattern = re.compile('(atc|info){1}_(code|type|repeated){1}\d{1}')
for col in subset.columns:
    if pattern.match(col) is not None:
        ac[col] = subset[col]
ac.to_csv('outputs/ASRS 1988-2019_extracted.csv', index = False)
