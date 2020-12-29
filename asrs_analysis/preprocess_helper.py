"""
Some helper functions to preprocess ASRS dataset.
"""
import re
import os
from collections import Counter

from tqdm import tqdm

import pandas as pd
import numpy as np

start_path = os.getcwd()
start_path_split = start_path.split("/")
if start_path_split[-1] != 'asrs_analysis':
    if os.path.exists(f'{start_path}/asrs_analysis'):
        start_path += '/asrs_analysis'
    else:
        raise ValueError(f"wrong cwd {start_path}")

def generate_ctrs_for_row(row):
    """
    This calculates the number of times each code appears in the same row (SFO),
    as well as the number of times each code type appears (TRACON/Tower/etc.)
    @param: row (pd.Series) from ASRS dataset
    @returns: code_ctr (collections.Counter) counts the number of times each code appears
    @returns: ident_ctr (collections.Counter) counts the number of times each ident appears
    @returns: num_na (int) the number of times either the code or the type were na
    """
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
    return code_ctr, ident_ctr, num_na

def add_duplicated_rows(row, code_ctr, ident_ctr, all_pds, dropcols):
    """
    This does the actual work of duplicating based off of tracon_code. It takes a row
    and the results from generate_ctrs_for_rows, and adds the duplicated rows to all_pds
    @param: row (pd.Series) from ASRS dataset
    @param: code_ctr (collections.Counter) counts the number of times each code appears
        in the row
    @param: ident_ctr (collections.Counter) counts the number of times each ident appears
        in the row
    @param: all_pds (list of pd.Series) final result, we add to this list
    @param: dropcols (list of str), which columns to drop when we construct the new dataframe
    """
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

                copy_row.drop(dropcols, axis=0, inplace=True)
                all_pds.append(copy_row)

                curr_codes.add(tracon_code)

def tracon_analysis(asrs):
    """
    In the ASRS dataset, each row could be attached to multiple tracons. This function
    duplicates these rows (one duplication for each tracon that is attached to the row)

    Hard duplicates occur when TRACON SFO TRACON SFO occur multiple times in the same row
    Soft duplicates occur when TRACON SFO Tower SFO occurs (same tracon, but different type).
    We expand soft duplicates but not hard duplicates. The columns analyzed in this function are
    {atc|info}_code{0-6} (this is generated in preprocess_asrs)

    @param: asrs (pd.DataFrame) ASRS dataset
    @returns: duplicated ASRS dataset (pd.DataFrame)
    """
    pat = re.compile(r'(info|atc)_(code|type|repeated)\d')
    dropcols = [col for col in asrs if pat.match(col)]

    all_pds = []
    for _, row in tqdm(asrs.iterrows(), total=asrs.shape[0]):
        # calculate the number of times a tracon_code appears
        # in this particular observation (including soft duplicates)
        code_ctr, ident_ctr, num_na = generate_ctrs_for_row(row)

        code_ctr, ident_ctr = dict(code_ctr), dict(ident_ctr)
        if num_na == 12: # still add the row if all of the codes are NA
            copy_row = row.copy()
            for key in ident_ctr:
                copy_row[f'{key}_ident_ct'] = ident_ctr[key]
            copy_row.drop(dropcols, axis=0, inplace=True)
            all_pds.append(copy_row)
        else:
            add_duplicated_rows(row, code_ctr, ident_ctr, all_pds, dropcols)

    return pd.DataFrame.from_records(all_pds)

def load_asrs(path=f'{start_path}/datasets/ASRS 1988-2019_extracted.csv', load_saved=False, \
        test=False):
    """
    This loads the ASRS dataset and pre-processes it. This is only done once in the pipeline,
    and all other times, we simply load a saved version.
    @param: path (str) path to asrs dataset
    @param: load_saved (bool) whether or not we load the saved version
    @param: test (bool) whether or not we should subsample the ASRS dataset for testing
        purposes
    @returns: processed ASRS dataset.
    """
    if load_saved:
        return pd.read_csv(f'{start_path}/results/asrs_extracted_processed.csv')

    asrs = pd.read_csv(path)

    # Dropping Duplicates
    dup_cols = ['ACN', 'narrative_report1', 'narrative_report2', 'synopsis_report1'] + \
            ['callback_report1', 'callback_report2', 'Locale Reference']
    asrs_dropped = asrs.drop_duplicates(dup_cols)
    asrs = asrs_dropped

    if test:
        np.random.seed(42)
        asrs = asrs.loc[np.random.choice(asrs.index, 1000, replace=False), :].copy()

    # for simplicity's sake we create an empty synopsis_report2
    asrs['synopsis_report2'] = np.nan

    type_reports = ['narrative', 'callback', 'synopsis']
    # creating fields for multiple reports
    for type_report in type_reports:
        # whether or not the row has multiple reports (t/f)
        asrs[f'{type_report}_multiple_reports'] = asrs.apply(lambda row:\
                (not pd.isna(row[f'{type_report}_report1'])) and \
                (not pd.isna(row[f'{type_report}_report2'])), axis=1)
    # creating field for containing a callback report
    asrs["contains_callback"] = asrs.apply(lambda row: \
            (not pd.isna(row["callback_report1"])) or \
            (not pd.isna(row["callback_report2"])), axis=1)
    # create combined reports
    for type_report in type_reports:
        report1 = asrs[f'{type_report}_report1'].replace(np.nan, '')
        report2 = asrs[f'{type_report}_report2'].replace(np.nan, '')
        asrs[type_report] = report1 + " " + report2

        # preprocess text
        asrs[type_report] = asrs.apply(lambda x: ' '.join(convert_to_words(x, type_report)), axis=1)

        asrs[type_report] = asrs[type_report].str.lower()
        asrs[f'{type_report}_report1'] = asrs[f'{type_report}_report1'].str.lower()
        asrs[f'{type_report}_report2'] = asrs[f'{type_report}_report2'].replace(np.nan, '')\
                .str.lower()

        asrs[asrs[type_report].duplicated(keep=False)]\
                .to_csv(f'results/duplicated_{type_report}.csv')

    def generate_date_cols(asrs):
        asrs['year'] = asrs['Date'].apply(lambda x: int(x // 100) if not pd.isna(x) else np.nan)
        asrs['month'] = asrs['Date'].apply(lambda x: int(x % 100) if not pd.isna(x) else np.nan)
        return asrs.loc[asrs['year'] != 20, :].copy()

    asrs['narrative_synopsis_combined'] = asrs['narrative'] + ' ' + asrs['synopsis']
    asrs['combined'] = asrs['narrative'] + ' ' + asrs['callback'] + ' ' + \
            asrs['synopsis']
    asrs['combined'] = asrs['combined'].str.lower()

    asrs = tracon_analysis(asrs)
    asrs = generate_date_cols(asrs)
    asrs = asrs.loc[(asrs['year'] >= 1988) & (asrs['year'] < 2020)]
    asrs.to_csv(f'{start_path}/results/asrs_extracted_processed.csv')
    return asrs

def load_dictionaries():
    """
    This loads all of our known aviation dictionaries into dataframes. The final result
    is a dictionary that maps from aviation dictionary name to corresponding dataframe.
    @returns: dict[aviation_dictionary_name] -> pd.DataFrame of dictionary
    """
    # casa
    casa = pd.read_csv(f'{start_path}/dictionaries/CASA.csv') # Fullform4
    casa = casa[["acronym", "Fullform4"]].copy().rename({"Fullform4": "casa_fullform"}, axis=1)
    casa['acronym'] = casa['acronym'].str.replace(r'\(.+\)', '')

    # faa
    faa = pd.read_csv(f'{start_path}/dictionaries/FAA.csv') # Fullform1
    faa = faa[["acronym", "Fullform1"]].copy().rename({"Fullform1": "faa_fullform"}, axis=1)

    # iata_iaco
    iata_iaco = pd.read_csv(f'{start_path}/dictionaries/IATA_IACO.csv', encoding='cp437')
    iata_iaco = iata_iaco[["acronym", "Fullform5"]].copy().rename({"Fullform5": "iata_fullform"}, \
            axis=1)
    iata_iaco = iata_iaco[~iata_iaco['acronym'].isna()]

    # nasa
    nasa = pd.read_csv(f'{start_path}/dictionaries/nasa_abbr.csv') # Fullform7
    nasa = nasa[["acronym", "Fullform7"]].copy().rename({"Fullform7": "nasa_fullform"}, axis=1)
    nasa.drop_duplicates(["acronym"], inplace=True, keep='first')

    # hand_code
    hand_code = pd.read_csv(f'{start_path}/dictionaries/hand_code.csv')
    hand_code = hand_code[["acronym", "Fullform6"]].copy().rename({"Fullform6": "hand_fullform"}, \
            axis=1)
    hand_code.drop_duplicates(["acronym"], inplace=True, keep='first')

    # hand_code2
    hand_code2 = pd.read_csv(f'{start_path}/dictionaries/combined_neg_nonword_handcode2.csv', \
            index_col=0)
    hand_code2.index.rename('acronym', inplace=True)
    hand_code2.reset_index(inplace=True)
    hand_code2.rename({'hand_fullform2': 'hand2_fullform'}, axis=1, inplace=True)
    hand_code2 = hand_code2[['acronym', 'hand2_fullform']]
    hand_code2.drop_duplicates(['acronym'], inplace=True, keep='first')
    hand_code2.dropna(axis=0, how='any', inplace=True)

    return {
        'casa': casa,
        'faa': faa,
        'iata': iata_iaco,
        'nasa': nasa,
        'hand': hand_code,
        'hand2': hand_code2
    }


def neg_nonword_to_neg_word_set():
    """
    This returns all neg_nonwords that are actually neg_words. Neg_nonwords are
    words that are not found in any aviation dictionary and are not in the english
    dictionary. We go through these by hand to find which ones are actually English words.
    @returns: set of neg nonwords that are actually english words
    """
    hand_code2 = pd.read_csv(f'{start_path}/dictionaries/combined_neg_nonword_handcode2.csv', \
            index_col=0)
    return set(hand_code2.loc[~hand_code2.loc[:, 'add_to_realworld_dictionary'].isna(), :].index)

def neg_nonword_to_airport_set():
    """
    This returns all neg_nonwords that are actually airport names. Neg_nonwords are
    words that are not found in any aviation dictionary and are not in the english
    dictionary. We go through these by hand to find which ones are actually names of
    airports
    @returns: set of neg nonwords that are actually airport names
    """
    hand_code2 = pd.read_csv(f'{start_path}/dictionaries/combined_neg_nonword_handcode2.csv', \
            index_col=0)
    return set(hand_code2.loc[~hand_code2.loc[:, 'add_to_airport'].isna(), :].index)

def neg_nonword_to_mispelled_dict():
    """
    This returns all neg_nonwords that are actually misspellings of english words. Neg_nonwords are
    words that are not found in any aviation dictionary and are not in the english
    dictionary. We go through these by hand to find which ones are actually misspellings
    of English words
    @returns: dictionary of neg nonwords that are actually airport names
    """
    hand_code2 = pd.read_csv(f'{start_path}/dictionaries/combined_neg_nonword_handcode2.csv',\
            index_col=0)
    return dict(hand_code2.loc[~hand_code2.loc[:, 'mispelled_word_fix'].isna(), \
            'mispelled_word_fix'])

def potential_words_from_negnw():
    """
    This creates a set of all neg_nonwords that we have not analyzed via hand-coded analysis
    @returns: set of all neg_nonwords we have not analyzed by hand
    """
    hand_code2 = pd.read_csv(f'{start_path}/dictionaries/combined_neg_nonword_handcode2.csv')

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

# the following are regex patterns that help us clean words

# starts with (, ' or [
R1 = r"^([\(\'\[]{1})([A-Za-z\d]{1,})$"
# ends with ), ' or ]
R2 = r"^([A-Za-z\d]{1,})([\)\'\]]{1})$"
# starts with (, ' or [ and ends with ), ', ]
R3 = r"^([\(\'\[\]]{1})([A-Za-z\d]{1,})([\(\'\[\]]{1})$"
# ends with ? or :, and only has alphabetical characters
R4 = r"^([A-Za-z]{1,})([\?:])$"
# only alphabetical with a / in the middle
R5 = r"^([A-Za-z]{1,})/([A-Za-z]{1,})$"

pats = [R3, R1, R2, R4, R5]
grps = [2, 2, 1, 1, 1] # which regex groups we need to extract
mispelled_dict = neg_nonword_to_mispelled_dict()

def clean_word(elem):
    """
    Given a word, we attempt to clean it utilizing the regex patterns
    seen above
    @param: elem (str) the word
    @returns: cleaned word or None if none of them match
    """
    for reg_pat, grp_idx in zip(pats, grps):
        pat_curr = re.compile(reg_pat)
        match_res = pat_curr.match(elem)
        if match_res is not None:
            return match_res.group(grp_idx)
    return None

def convert_to_words(row, col='narrative', replace_dict={}):
    """
    This converts a string or pandas.Series containing a string into
    a cleaned tokenizerd version of the string (in a np.ndarray)
    @param: row (str/pd.Series) containing the string we wish to analyze
    @param: col (str) column we are analyzing, only utilized if row
        is not a string
    @param: replace_dict (dict[word] -> replace_word), dictionary of
        words we wish to replace with another version (abbreviation -> fullform)
    @returns: np.ndarray of cleaned tokenized version of the string in question
    """
    if not isinstance(row, str):
        field = row[col]
    else:
        field = row
    if isinstance(field, float) and np.isnan(field):
        field = ''
    for char in r'[!"#$&\'()*+,:;<=>?@[\\]^_`{|}~/-]\.%':
        field = field.replace(char, ' ')
    res = np.array(re.split(r'[( | ;|\. |\.$]', field))
    res = res[res != ''].flatten()
    fin = []
    for elem in res:
        if elem in replace_dict:
            fin.append(replace_dict[elem])
        else:
            cleaned_word = clean_word(elem)
            if cleaned_word is None:
                cleaned_word = elem
            fin.append(elem)
    return np.array(fin)

def generator_split(split_series):
    """
    Helper function that iterates through a nested iterable, and yields
    each word
    """
    for list_elem in split_series:
        for word in list_elem:
            yield word

def replace_words(sentence, replace_dict={}):
    """
    Utilize replace_dict to replace words in a given sentence. For example,
    replace_words('hello jimmy', {'hello':'bye'}) = 'bye jimmy'
    @param: sentence (str) sentence we wish to process
    @param: replace_dict (dictionary of strings) dictionary mapping from old
        words to new words
    @returns: np.ndarray of tokenized words in the sentence
    """
    if isinstance(sentence, float) and pd.isna(sentence):
        return np.array([])
    split_sentence = sentence.split()
    for idx, elem in enumerate(split_sentence):
        split_sentence[idx] = replace_dict.get(elem, elem)
    return np.array(split_sentence)

def create_counter(text_df, col='narrative'):
    """
    This creates a dataframe that maps each unique word to the number of times
    it appears in the full dataset.
    @param: text_df (pd.DataFrame) text dataset
    @param: col (str) which column to analyze
    @returns: pd.DataFrame that maps unique words to the number of times they appear
        in the full dataset
    """
    tqdm.pandas(desc=col)

    dropped = text_df.drop_duplicates(col)
    split = dropped.progress_apply(lambda x: np.array(x[col].split()), axis=1)

    res = Counter(generator_split(split))
    res = pd.DataFrame.from_dict(dict(res), orient='index')
    return res
