import pandas as pd, re, numpy as np
from IPython import embed
from collections import Counter, namedtuple
from tqdm import tqdm

coverage = namedtuple('coverage', ['part', 'total'])

def load_asrs(path = 'datasets/ASRS 1988-2019_extracted.csv', load_saved = False):
    if load_saved:
        return pd.read_csv('results/asrs_extracted_processed.csv')

    asrs = pd.read_csv(path)

    # Dropping Duplicates
    dup_cols = ['ACN', 'narrative_report1', 'narrative_report2', 'synopsis_report1'] + \
            ['callback_report1', 'callback_report2', 'Locale Reference']
    asrs_dropped = asrs.drop_duplicates(dup_cols)
    # asrs_dropped = asrs.drop_duplicates()
    print(coverage(part = asrs_dropped.shape[0], \
            total = asrs.shape[0]))
    asrs = asrs_dropped

    # for simplicity's sake we create an empty synopsis_report2
    asrs['synopsis_report2'] = np.nan

    type_reports = ['narrative', 'callback', 'synopsis']
    # creating fields for multiple reports
    for type_report in type_reports:
        # whether or not the row has multiple reports (t/f)
        asrs[f'{type_report}_multiple_reports'] = asrs.apply(lambda row: \
                (not pd.isna(row[f'{type_report}_report1'])) and
                (not pd.isna(row[f'{type_report}_report2'])), axis = 1
        )
    # creating field for containing a callback report
    asrs["contains_callback"] = asrs.apply(lambda row: \
            (not pd.isna(row["callback_report1"])) or
            (not pd.isna(row["callback_report2"])), axis = 1
    )
    # create combined reports
    for type_report in type_reports:
        report1 = asrs[f'{type_report}_report1'].replace(np.nan, '')
        report2 = asrs[f'{type_report}_report2'].replace(np.nan, '')
        asrs[type_report] = report1 + " " + report2

    def generate_date_cols(asrs):
        asrs['year'] = asrs['Date'].apply(lambda x: int(x // 100) if not pd.isna(x) else np.nan)
        asrs['month'] = asrs['Date'].apply(lambda x: int(x % 100) if not pd.isna(x) else np.nan)
        return asrs.loc[asrs['year'] != 20, :].copy()

    def tracon_analysis(asrs):
        pat = re.compile('(info|atc)_(code|type|repeated)\d')
        dropcols = [col for col in asrs if pat.match(col)]

        all_pds = []
        for idx, row in tqdm(asrs.iterrows(), total = asrs.shape[0]):

            # calculate the number of times a tracon_code appears
            # in this particular observation (including soft duplicates)
            code_ctr, ident_ctr = Counter(), Counter()
            code_and_type_set = set()
            num_na = 0
            for info_type in ['atc', 'info']:
                for i in range(6):
                    tracon_code = row[f'{info_type}_code{i}']
                    tracon_type = row[f'{info_type}_type{i}']

                    if pd.isna(tracon_code) and pd.isna(tracon_type):
                        num_na += 1
                    else:
                        code_and_type = f'{tracon_code} {tracon_type}'

                        # add soft-duplicates to code_ctr but exclude hard-duplicates
                        if code_and_type not in code_and_type_set:
                            code_ctr.update([tracon_code])
                            ident_ctr.update([tracon_type])

                        code_and_type_set.add(code_and_type)
            code_ctr, ident_ctr = dict(code_ctr), dict(ident_ctr)
            if num_na == 12:
                # still add the row if all of the codes are NA
                copy_row = row.copy()
                for key in ident_ctr:
                    copy_row[f'{key}_ident_ct'] = ident_ctr[key]
                copy_row.drop(dropcols, axis = 0, inplace = True)
                all_pds.append(copy_row)
            else:
                curr_codes = set()
                for info_type in ['atc', 'info']:
                    for i in range(6):
                        tracon_code = row[f'{info_type}_code{i}']
                        if tracon_code not in curr_codes and not pd.isna(tracon_code):
                            copy_row = row.copy()
                            copy_row['tracon_code'] = tracon_code
                            copy_row['ident_type'] = row[f'{info_type}_type{i}']
                            copy_row['num_code_per_obs'] = code_ctr[tracon_code]
                            # add number of each ident to the row
                            for key in ident_ctr:
                                copy_row[f'{key}_ident_ct'] = ident_ctr[key]

                            copy_row.drop(dropcols, axis = 0, inplace = True)
                            all_pds.append(copy_row)

                            curr_codes.add(tracon_code)
        all_pds = pd.DataFrame.from_records(all_pds)
        return all_pds

    asrs['narrative_synopsis_combined'] = asrs['narrative'] + ' ' + asrs['synopsis']
    asrs['combined'] = asrs['narrative'] + ' ' + asrs['callback'] + ' ' + \
            asrs['synopsis']

    cols = ['narrative', 'synopsis', 'callback']
    for col in cols:
        asrs[col] = asrs[col].str.lower()
        asrs[f'{col}_report1'] = asrs[f'{col}_report1'].str.lower()
        asrs[f'{col}_report2'] = asrs[f'{col}_report2'].replace(np.nan, '').str.lower()
    asrs['combined'] = asrs['combined'].str.lower()

    # TODO: remove this, only doing this so it doesn't destroy my laptop
    # np.random.seed(42)
    # asrs = asrs.loc[np.random.choice(asrs.index, 1000), :].copy()

    total = asrs.shape[0]
    asrs = tracon_analysis(asrs)
    print(coverage(part = asrs.shape[0], total = total))
    asrs = generate_date_cols(asrs)
    total = asrs.shape[0]
    asrs = asrs.loc[(asrs['year'] >= 1988) & (asrs['year'] < 2020)]
    print(coverage(asrs.shape[0], total))
    asrs.to_csv('results/asrs_extracted_processed.csv')
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

    # hand_code2
    hand_code2 = pd.read_csv('dictionaries/combined_neg_nonword_handcode2.csv', index_col = 0)
    hand_code2.index.rename('acronym', inplace = True)
    hand_code2.reset_index(inplace = True)
    hand_code2.rename({'hand_fullform2': 'hand2_fullform'}, axis = 1, inplace = True)
    hand_code2 = hand_code2[['acronym', 'hand2_fullform']]
    hand_code2.drop_duplicates(['acronym'], inplace = True, keep = 'first')
    hand_code2.dropna(axis = 0, how = 'any', inplace = True)

    return {
        'casa': casa,
        'faa': faa,
        'iata': iata_iaco,
        'nasa': nasa,
        'hand': hand_code,
        'hand2': hand_code2
    }


def neg_nonword_to_neg_word_set():
    hand_code2 = pd.read_csv('dictionaries/combined_neg_nonword_handcode2.csv', index_col = 0)
    return set(hand_code2.loc[~hand_code2.loc[:, 'add_to_realworld_dictionary'].isna(), :].index)

def neg_nonword_to_airport_set():
    hand_code2 = pd.read_csv('dictionaries/combined_neg_nonword_handcode2.csv', index_col = 0)
    return set(hand_code2.loc[~hand_code2.loc[:, 'add_to_airport'].isna(), :].index)

def neg_nonword_to_mispelled_dict():
    hand_code2 = pd.read_csv('dictionaries/combined_neg_nonword_handcode2.csv', index_col = 0)
    return dict(hand_code2.loc[~hand_code2.loc[:, 'mispelled_word_fix'].isna(), 'mispelled_word_fix'])

def potential_words_from_negnw():
    hand_code2 = pd.read_csv('dictionaries/combined_neg_nonword_handcode2.csv')

    # calculate last element with any of these columns
    added_cols = ['hand_fullform2', 'add_to_realworld_dictionary', 'add_to_airport', \
            'mispelled_word_fix', 'Notes for future']
    last_idx = 0
    for col in added_cols:
        non_na = ~hand_code2[col].isna()
        idx = non_na[::-1].idxmax()
        if idx > last_idx:
            last_idx = idx
    return set(hand_code2.loc[hand_code2.index > last_idx, 'word'])



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
mispelled_dict = neg_nonword_to_mispelled_dict()

def convert_to_words(row, col = 'narrative', replace_dict = {}):
    # replace_dict.update(mispelled_dict)
    s = row[col]
    if isinstance(s, float) and np.isnan(s):
        s = ''
    for char in '[!"#$&\'()*+,:;<=>?@[\\]^_`{|}~/-]\.%':
        s = s.replace(char, ' ')
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
    tqdm.pandas(desc = col)
    # split = df.apply(lambda x: convert_to_words(x, col, mispelled_dict), axis = 1)
    dropped = df.drop_duplicates(col)
    split = dropped.progress_apply(lambda x: convert_to_words(x, col, mispelled_dict), axis = 1)
    res = Counter(generator_split(split))
    res = pd.DataFrame.from_dict(dict(res), orient = 'index')
    return res
