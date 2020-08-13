import pandas as pd, numpy as np, matplotlib.pyplot as plt
import re

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
ac = pd.read_csv('datasets/ASRS 1988-2019.csv')

# create sets
wiki_pd = pd.read_csv('datasets/wiki_code.csv')
tracon_pd = pd.read_csv('datasets/tracon_code.csv')

all_codes = set(wiki_pd['IATA'])
all_codes.update(wiki_pd['ICAO'])
all_codes.update(tracon_pd['LocID'])
all_codes.remove(np.nan)

# remove all np nans and turn them into empty strings
ac.loc[ac['atcadvisory'].isnull(), 'atcadvisory'] = ''

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

# Deal with some leftover garbage (only TRACON/Tower left), this only finds 2 rows
d = set(['TRACON', 'Tower', ''])
ct = 0
all_idx = []
for row in subset['removed_firstpass'].iteritems():
    all_d = True
    for elem in row[1].split(' '):
        if elem not in d:
            all_d = False
    if all_d:
        ct += 1
        all_idx.append(row[0])
subset.loc[all_idx, 'removed_firstpass'] = '' 
leftovers = subset[~subset['removed_firstpass'].str.match(r'^ *$')]
leftovers.to_csv('outputs/leftovers.csv')

# MANUAL FIXING. Check leftovers.csv if you need to double check
fixed_indices = [10989, 22166, 40340, 45628, 54368, 83361, 84230, \
        90534, 90826, 91426, 91672, 97163, 97242, 98013, 101283, \
        106488, 108336, 130642, 132420, 139763, 162464, 210483, 225008]
subset.loc[10989, 'firstpass'] = [('TRACON', 'O', '')]
# 21879 has miscellaneous ampersand
subset.loc[22166, 'firstpass'].append(('Tower', 'V. C. BIRD', '')) # BIRD not in set anyways
subset.loc[40340, 'firstpass'] = [('Military Facility', 'USAF ACFT', '')] + \
        subset.loc[40340, 'firstpass'][1:] # USAF and ACFT are not in set 
subset.loc[45628, 'firstpass'] = [('TRACON', 'N(0', '')]
subset.loc[54368, 'firstpass'] = subset.loc[54368, 'firstpass'][:-1] + \
        [('Military Facility', 'RIDGE CTRL', '')] # RIDGE/CTRL not found 
subset.loc[83361, 'firstpass'] = subset.loc[83361, 'firstpass'][:1] + \
        [('Military Facility', 'ZBW CAPE', '')] + subset.loc[83361, 'firstpass'][2:]
subset.loc[84230, 'firstpass'] = subset.loc[84230, 'firstpass'][:-1] + \
        [('FSS', 'JNU', ''), ('FSS', 'RDU', '')] # Both of these are found, so separated
subset.loc[90534, 'firstpass'] = subset.loc[90534, 'firstpass'][:-1] + \
        [('UNICOM', '998.00', '')]
subset.loc[90826, 'firstpass'] = subset.loc[90826, 'firstpass'][:-1] + \
        [('UNICOM', '123.14', '')]
subset.loc[91426, 'firstpass'] = subset.loc[91426, 'firstpass'][:2] + \
        [('UNICOM', '998.00', ''), ('UNICOM', '159', '')]
subset.loc[91672, 'firstpass'] = subset.loc[91672, 'firstpass'][:2] + \
        [('UNICOM', '123.14', ''), ('UNICOM', '159', '')]
# 91672 is skipped b/c LAX is in sets, and 'Special Flight Rules' is excess text
# 97079 same thing
subset.loc[97163, 'firstpass'] = [('CTAF', 'Airshow Control', '')]
subset.loc[97242, 'firstpass'] = [('UNICOM', 'GROVE CITY', '')]
subset.loc[98013, 'firstpass'] = subset.loc[98013, 'firstpass'][:2] + \
        [('UNICOM', '148.30', ''), ('UNICOM', '159', '')]
# 99991 kept excess info in removed_firstpass because code was found
subset.loc[101283, 'firstpass'] = subset.loc[101283, 'firstpass'][:-1] + \
        [('UNICOM', 'Local FBO', '')] # Local/FBO not found
subset.loc[106488, 'firstpass'] =  subset.loc[106488, 'firstpass'] + [('CTAF', '', '')]
subset.loc[108336, 'firstpass'] =  [('UNICOM', 'I-23', '')] # not found
subset.loc[130642, 'firstpass'] =  [('CTAF', 'OTX 1', '')] # not found
subset.loc[132420, 'firstpass'].append(('UNICOM', '0', '')) # not found
subset.loc[139763, 'firstpass'].append(('CTAF', 'NOT', '')) # not found
# only semicolon, leaving it
subset.loc[162464, 'firstpass'] = [('UNICOM', 'BAY BRIDGE', '')] # BAY is found
# but that's located in romania (not the BAY BRIDGE)
subset.loc[210483, 'firstpass'].append(('CTAF', '', ''))
subset.loc[225008, 'firstpass'] = [('UNICOM', '122.9', '')]
subset.loc[fixed_indices, 'remove_firstpass'] = ''

subset = separate_nonatc(subset)
subset = expand_atc(subset)

# Create dictionaries of the most common info codes and types
# key -> value: code -> number of times it is used
code_ct_dict, type_ct_dict = {}, {}
for i in range(4):
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
