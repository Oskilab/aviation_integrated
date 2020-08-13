import pandas as pd, re, numpy as np
from IPython import embed
from collections import Counter

def load_asrs(path = 'datasets/ASRS 1988-2019_extracted.csv'):
    asrs = pd.read_csv(path)
    def generate_date_cols(asrs):
        asrs['year'] = asrs['date'].apply(lambda x: int(x // 100))
        asrs['month'] = asrs['date'].apply(lambda x: int(x % 100))
        asrs['quarter'] = (asrs['month'] - 1) // 3 + 1
        return asrs.loc[asrs['year'] != 20, :].copy()
    def tracon_analysis(asrs):
        pat = re.compile('(info|atc)_(code|type|repeated)\d')
        dropcols = [col for col in asrs if pat.match(col)]

        all_pds = []
        for info_type in ['atc', 'info']:
            for i in range(4):
                curr = asrs.loc[asrs[f"{info_type}_type{i}"] == 'TRACON', :].copy()
                tracon_code = curr[f"{info_type}_code{i}"].copy()
                curr.drop(dropcols, axis = 1, inplace = True)
                curr['tracon_code'] = tracon_code
                all_pds.append(curr)
        all_pds = pd.concat(all_pds, ignore_index = True)
        return all_pds

    tolist = lambda x: [x['synopsis'], x['narrative']]
    asrs['combined'] = asrs.apply(tolist, axis = 1).str.join(sep = ' ')
    cols = ['narrative', 'synopsis', 'combined']
    for col in cols:
        asrs[col] = asrs[col].str.lower()
    asrs = tracon_analysis(asrs)
    asrs = generate_date_cols(asrs)
    return asrs

def load_dictionaries():
    # casa
    casa = pd.read_csv('dictionaries/CASA.csv') # Fullform4
    casa = casa[["acronym", "Fullform4"]].copy().rename({"Fullform4": "casa_fullform"}, axis = 1)
    casa['acronym'] = casa['acronym'].str.replace('\(.+\)', '')

    # faa
    faa = pd.read_csv('dictionaries/FAA.csv') # Fullform1
    faa = faa[["acronym", "Fullform1"]].copy().rename({"Fullform1": "faa_fullform"}, axis = 1)

    # iata_iaco
    iata_iaco = pd.read_csv('dictionaries/IATA_IACO.csv', encoding = 'cp437') # Fullform5
    iata_iaco = iata_iaco[["acronym", "Fullform5"]].copy().rename({"Fullform5": "iata_fullform"}, axis = 1)
    iata_iaco = iata_iaco[~iata_iaco['acronym'].isna()]

    # nasa
    nasa = pd.read_csv('dictionaries/nasa_abbr.csv') # Fullform7
    nasa = nasa[["acronym", "Fullform7"]].copy().rename({"Fullform7": "nasa_fullform"}, axis = 1)
    nasa.drop_duplicates(["acronym"], inplace = True, keep = 'first')

    # hand_code
    hand_code = pd.read_csv('dictionaries/hand_code.csv')
    hand_code = hand_code[["acronym", "Fullform6"]].copy().rename({"Fullform6": "hand_fullform"}, axis = 1)
    hand_code.drop_duplicates(["acronym"], inplace = True, keep = 'first')

    return {
        'casa': casa,
        'faa': faa,
        'iata': iata_iaco,
        'nasa': nasa,
        'hand': hand_code
    }


# starts with (, ' or [
r1 = "^([\(\'\[]{1})([A-Za-z\d]{1,})$"
# ends with ), ' or ]
r2 = "^([A-Za-z\d]{1,})([\)\'\]]{1})$"
# starts with (, ' or [ and ends with ), ', ]
r3 = "^([\(\'\[\]]{1})([A-Za-z\d]{1,})([\(\'\[\]]{1})$"
# ends with ? or :, and only has alphabetical characters
r4 = "^([A-Za-z]{1,})([\?:])$"
# only alphabetical with a / in the middle
r5 = "^([A-Za-z]{1,})/([A-Za-z]{1,})$"

pats = [r3, r1, r2, r4, r5]
grps = [2, 2, 1, 1, 1]

def convert_to_words(row, col = 'narrative', replace_dict = {}):
    s = row[col]
    if isinstance(s, float) and np.isnan(s):
        s = ''
    res = np.array(re.split('[( | ;|\. |\.$]', s))
    res = res[res != ''].flatten()
    fin = []
    for elem in res:
        if elem in replace_dict:
            fin.append(replace_dict[elem])
        else:
            added = False
            for r, grp_idx in zip(pats, grps):
                pat_curr = re.compile(r)
                match_res = pat_curr.match(elem)
                if match_res is not None:
                    fin.append(match_res.group(grp_idx))
                    added = added or match_res is not None
                    break
            if not added:
                fin.append(elem)
    return np.array(fin)

def generator_split(split_series):
    for list_elem in split_series:
        for x in list_elem:
            yield x

def create_counter(df, col = 'narrative'):
    print('create_counter')
    split = df.apply(lambda x: convert_to_words(x, col), axis = 1)
    res = Counter(generator_split(split))
    res = pd.DataFrame.from_dict(dict(res), orient = 'index')
    return res

# asrs = load_asrs()
# res = create_counter(asrs, col = 'narrative')
# embed()
