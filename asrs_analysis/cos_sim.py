from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython import embed
from collections import Counter
from tqdm import tqdm
from preprocess_helper import *
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import pandas as pd, numpy as np, re, pickle, argparse

parser = argparse.ArgumentParser(description='Analyze abbreviations.')
parser.add_argument('-t', action = 'store_true')
args = parser.parse_args()

test = args.t
num_time_periods = (2020 - 1988) * 12
all_pds = load_asrs(load_saved = False)

def num_months_between(month1, year1, month2, year2):
    return (year2 - year1) * 12 + month2 - month1

def generate_compare(month1, year1, num_months = 1): # accident date
    """
    This returns a function to be applied row-wise on a dataframe. The inner function, when
    applied, returns a pandas series that finds which rows are within a month range. If
    the FAA/NTSB incident tracon_month occurs in January 2011, then the inner function finds
    all the rows that are within num_months months before January 2011.
    @param: month1 (int) 1-12
    @param: year1 (int)
    @param: num_months (int)
    @returns: function(row) that returns a pandas Series selecting the correct rows
    """
    def inner_func(row):
        month2, year2 = row['month'], row['year']
        n_m = num_months_between(month1, year1, month2, year2)
        return n_m > 0 and n_m <= num_months
    return inner_func
def calculate_avg_comp(list_idx1, list_idx2, d2v_model, overlap = 0, same = False):
    if len(list_idx1) == 1 and len(list_idx2) == 1 and same:
        return np.nan, np.nan
    elif len(list_idx1) > 0 and len(list_idx2) > 0:
        d2v1 = np.vstack([d2v_model.docvecs[x] for x in list_idx1])
        d2v2 = np.vstack([d2v_model.docvecs[x] for x in list_idx2])

        cos_res = cosine_similarity(d2v1, d2v2)
        sum_d2v = np.sum(cos_res) - overlap
        num_comp = cos_res.shape[0] * cos_res.shape[1] - overlap
        # just a test
        assert(cos_res.shape[0] == len(list_idx1) and cos_res.shape[1] == len(list_idx2))
        avg_d2v = sum_d2v / num_comp
        if same:
            num_comp /= 2 # you're comparing to itself so 2way symmetry
        return avg_d2v, num_comp
    else:
        return np.nan, np.nan
def calculate_avg_comp2(list_idx1, list_idx2, cos_res, overlap = 0, same = False):
    """
    Average cosine similarity function utilizing indices + matrix (see below).
    @param: list_idx1 (pd.index) of indices into cos_res
    @param: list_idx2 (pd.index) of indices into cos_res
    @param: cos_res (np.ndarray) of pairwise cosine similarity metric, must be square
    @returns: (avg_d2v, num_comp)
        avg_d2v = average cosine similarity with a row index within list_idx2 and a 
            column index within list_idx2,
        num_comp = number of comparisons made
    """
    num_comp = len(list_idx1) * len(list_idx2) - overlap
    if num_comp <= 0:
        return np.nan, np.nan
    else:
        list_idx1, list_idx2 = np.array(list_idx1), np.array(list_idx2)
        sum_d2v = np.sum(cos_res[list_idx1[:, None], list_idx2])
        avg_d2v = sum_d2v / num_comp
        if same:
            num_comp /= 2
        return avg_d2v, num_comp
        

abrev_col_dict = {'narrative': 'narr', 'synopsis': 'syn', \
        'narrative_synopsis_combined': 'narrsyn', 'combined': 'all', \
        'callback': 'call'}
def analyze_d2v(all_pds, d2v_model, replace = True, month_range_dict = {}, col = ""):
    """
    @param: all_pds(pd.DataFrame) should be the full asrs dataset
    @param: d2v_model (gensim.models.doc2vec) model that was trained on full dataset
    @param: replace (bool): if true, the abbreviations were replaced by fullforms
    @param: month_range_dict (dict): month_range (1/3/6/12/inf) -> list of dataframes
        where each dataframe has the relevant doc2vec comparison info
    """
    abrev_col = abrev_col_dict[col]
    for month_range in [1, 3, 6, 12]:
        mr_str = str(month_range)
        if month_range == np.inf:
            mr_str = 'a'
        mr_str += 'm'

        # used in d2v column names
        end_str = f"{abrev_col}_{mr_str}" 
        col_type1 = f'{"_flfrm" if replace else ""}_{end_str}'
        col_type2 = f'{"_flfrm" if replace else ""}_ct_{end_str}'

        # generate dictionaries for caching
        tracon_month_dict, cos_res_tracon_dict = {}, {}
        for month, year in tqdm(product(range(1, 13), range(1988, 2020)), total = num_time_periods, \
                desc = f"{col} dict creation {month_range}mon"):
            code = ' '.join([str(month), str(year)])

            # select only the rows within the month range
            compare_func = generate_compare(month, year, num_months = month_range)
            tracon_month_dict[code] = all_pds.loc[all_pds.apply(compare_func, axis = 1), :].copy()

            all_tracon = list(tracon_month_dict[code].drop_duplicates(col).index)

            if len(all_tracon) >= 1:
                d2v1 = np.vstack([d2v_model.docvecs[x] for x in all_tracon])
                cos_res_tracon_dict[code] = cosine_similarity(d2v1, d2v1)
            else:
                cos_res_tracon_dict[code] = np.zeros((0, 0))

        index_to_d2v  = {}

        # actually generate d2v cosine analysis data
        for idx, row in tqdm(tracon_month_unique.iterrows(), total = tracon_month_unique.shape[0], \
                desc = f"{col} analysis {month_range}mon"):
            index_id = f"{row['tracon_code']} {row['year']}/{row['month']}"
            code = ' '.join([str(row['month']), str(row['year'])])

            # if code in cos_res_tracon_dict:
            cos_res = cos_res_tracon_dict[code]
            searched = tracon_month_dict[code] # all rows with time period
            re_indexed_search = searched.drop_duplicates(col).reset_index()
            same_tracon = re_indexed_search.loc[ \
                    re_indexed_search['tracon_code'] == row['tracon_code'], :].index
            other_tracon = re_indexed_search.loc[ \
                    re_indexed_search['tracon_code'] != row['tracon_code'], :].index
            all_tracon = re_indexed_search.index

            d2v_dict = {}

            # same to same tracon
            avg_d2v, num_comp = calculate_avg_comp2(same_tracon, same_tracon, cos_res, \
                    overlap = len(same_tracon), same = True)
            d2v_dict[f'trcn{col_type1}'] = avg_d2v
            d2v_dict[f'trcn{col_type2}'] = num_comp

            # same to other tracon
            avg_d2v, num_comp = calculate_avg_comp2(same_tracon, other_tracon, cos_res)
            d2v_dict[f'trcn_invout{col_type1}'] = avg_d2v
            d2v_dict[f'trcn_invout{col_type2}'] = num_comp

            # other to other tracon
            avg_d2v, num_comp = calculate_avg_comp2(other_tracon, other_tracon, cos_res, \
                    overlap = len(other_tracon), same = True)
            d2v_dict[f'trcn_out{col_type1}'] = avg_d2v
            d2v_dict[f'trcn_out{col_type2}'] = num_comp

            # same to all tracon
            avg_d2v, num_comp = calculate_avg_comp2(same_tracon, all_tracon, cos_res, \
                    overlap = len(same_tracon))
            d2v_dict[f'trcn_invall{col_type1}'] = avg_d2v
            d2v_dict[f'trcn_invall{col_type2}'] = num_comp

            # all to all tracon
            avg_d2v, num_comp = calculate_avg_comp2(all_tracon, all_tracon, cos_res, \
                    overlap = len(all_tracon), same = True)
            d2v_dict[f'trcn_all{col_type1}'] = avg_d2v
            d2v_dict[f'trcn_all{col_type2}'] = num_comp

            # b/w report1 and report2
            if col == 'narrative' or col == 'callback': # only those with mult reports
                this_tracon = searched.loc[searched['tracon_code'] == row['tracon_code'], :].copy()
                this_tracon.drop_duplicates([f'{col}_report1', f'{col}_report2'], inplace = True)
                d2v_dict[f'trcn_mult_{abrev_col}{"_flfrm" if replace else ""}'] = \
                        this_tracon[f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}'].mean()
                d2v_dict[f'trcn_mult_{abrev_col}{"_flfrm" if replace else ""}_ct'] = \
                        this_tracon[f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}'].count()

            index_to_d2v[index_id] = pd.Series(d2v_dict)
        fin = pd.DataFrame.from_dict(index_to_d2v, orient = 'index')
        month_range_dict[month_range] = month_range_dict.get(month_range, []) + [fin]

def load_replace_dictionary(col):
    """
    @param: col (str): narrative/synopsis/combined/narrative_synopsis_combined
        which string we are currently analyzing
    @returns: replace_dict (dict): abbreviation -> fullform
    """
    # TODO: put this in preprocess_helper.py
    total_cts = pd.read_csv(f'results/total_cts_tagged_{col}.csv', index_col = 0)
    total_cts = total_cts.loc[total_cts['abrev'] == 1, :]

    # generate replace dictionary
    replace_dict = {}
    for idx, row in total_cts.iterrows():
        for dict_name in dictionary_names:
            dict_fullform = row[f'{dict_name}_fullform']
            if not pd.isna(dict_fullform):
                dict_fullform = str(dict_fullform).lower()
                replace_dict[row['acronym']] = dict_fullform
                break
    return replace_dict

def generate_duplicated_idx(all_pds, field, mult_col, report_num = 1):
    dup_idx = [f'{index} 1' for index in \
            all_pds.loc[all_pds[f'{mult_col}_report1'] == field, :].index]
    dup_idx += [f'{index} 2' for index in \
            all_pds.loc[all_pds[f'{mult_col}_report2'] == field, :].index]
    return dup_idx

dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'iata']
all_pds = all_pds.reset_index().drop('index', axis = 1)

tracon_month_unique = all_pds[['tracon_code', 'month', 'year']].drop_duplicates()
all_combs = set(tracon_month_unique.apply(lambda x: (x[0], x[1], x[2]), axis = 1))

# TODO: check that this pickle file is automatically being generated in pipeline
unique_code_fn = '../results/unique_airport_code_ntsb_faa.pckl'
unique_ntsb_faa_codes = pickle.load(open(unique_code_fn, 'rb'))
unique_codes = set(unique_ntsb_faa_codes)

asrs_added_tracons = []
for tracon_code in tracon_month_unique['tracon_code'].unique():
    if tracon_code not in unique_codes:
        asrs_added_tracons.append(tracon_code)

unique_ntsb_faa_codes = np.hstack([unique_ntsb_faa_codes, np.array(asrs_added_tracons)])

# if test:
# if we're testing only utilize the airport codes from this wikipedia file
top_50_iata = set(pd.read_excel('../datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
unique_ntsb_faa_codes = np.apply_along_axis(lambda x: [elem for elem in x if elem in top_50_iata], \
        0, unique_ntsb_faa_codes)
tracon_month_unique = tracon_month_unique.loc[\
    tracon_month_unique['tracon_code'].apply(lambda x: x in top_50_iata)]

added_rows = {'tracon_code': [], 'month': [], 'year':[]}
for tracon, month, year in tqdm(product(unique_ntsb_faa_codes, range(1, 13), range(1988, 2020)), \
        desc = 'adding empty rows', total = num_time_periods * unique_ntsb_faa_codes.shape[0]):
    if (tracon, month, year) not in all_combs:
        added_rows['tracon_code'].append(tracon)
        added_rows['month'].append(month)
        added_rows['year'].append(year)

tracon_month_unique = tracon_month_unique.append(pd.DataFrame.from_dict(added_rows))

# all_pds = all_pds.loc[all_pds['tracon_code'].apply(lambda x: x in top_50_iata)]
# deal with multiple reports
for mult_col in ['narrative', 'callback']:
    for r_d in [load_replace_dictionary(mult_col), {}]:
        replace = len(r_d) > 0
        cos_col_name = f'{mult_col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}'
        all_pds[cos_col_name] = np.nan

        # creating list of tagged documents
        docs, set_of_docs = [], set()
        for idx, row in tqdm(all_pds.iterrows(), total = all_pds.shape[0], desc = 'creating' + \
                f' documents for {mult_col}{" replace" if replace else ""}'):
            # note duplicated code, TODO: fix
            field1 = row[f'{mult_col}_report1']
            if field1 not in set_of_docs and not pd.isna(field1):
                dup_idx = generate_duplicated_idx(all_pds, field1, mult_col, field1)
                doc_str = ' '.join(convert_to_words(row, f'{mult_col}_report1', r_d))
                docs.append(TaggedDocument(doc_str, dup_idx))
                set_of_docs.add(field1)

            field2 = row[f'{mult_col}_report2']
            if field2 not in set_of_docs and not pd.isna(field2):
                dup_idx = generate_duplicated_idx(all_pds, field2, mult_col, field2)
                doc_str = ' '.join(convert_to_words(row, f'{mult_col}_report2', r_d))
                docs.append(TaggedDocument(doc_str, dup_idx))
                set_of_docs.add(field2)

        # train doc2vec
        model = Doc2Vec(docs, vector_size = 20, window = 3)
        only_mult_rep_df = all_pds.loc[all_pds[f'{mult_col}_multiple_reports'], :]
        for idx, row in only_mult_rep_df.iterrows():
            if row[f'{mult_col}_report1'] == row[f'{mult_col}_report2']:
                vec1, vec2 = model.docvecs[f'{idx} 1'], model.docvecs[f'{idx} 1']
            else:
                vec1, vec2 = model.docvecs[f'{idx} 1'], model.docvecs[f'{idx} 2']
            cos_sim = cosine_similarity(vec1.reshape(1, 20), vec2.reshape(1, 20))
            all_pds.loc[idx, cos_col_name] = cos_sim[0, 0]


for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
    month_range_dict = {}
    for r_d in [load_replace_dictionary(col), {}]:
        # creating tagged documents
        docs, set_of_docs = [], set()
        for idx, row in all_pds.iterrows():
            if row[col] not in set_of_docs:
                # preprocess document
                np_res = convert_to_words(row, col, r_d)
                doc_str = ' '.join(np_res)

                all_idx = list(all_pds.loc[all_pds[col] == row[col]].index)
                docs.append(TaggedDocument(doc_str, all_idx))

                # to remove duplicates
                set_of_docs.add(row[col])

        # train doc2vec
        print('training doc2vec models. This can take a while...')
        model = Doc2Vec(docs, vector_size = 20, window = 3)

        analyze_d2v(all_pds, model, len(r_d) > 0, month_range_dict, col = col)

    for month_range in month_range_dict.keys():
        res = pd.concat(month_range_dict[month_range], axis = 1)
        res.to_csv(f'results/d2v_tracon_month_{col}_{month_range}mon.csv')
