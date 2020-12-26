from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from collections import Counter
from tqdm import tqdm
import time
# from asrs_analysis.preprocess_helper import *
from preprocess_helper import *
from sklearn.metrics.pairwise import cosine_similarity
from itertools import product
import pandas as pd, numpy as np, re, pickle, argparse
from itertools import chain

parser = argparse.ArgumentParser(description='Analyze abbreviations.')
parser.add_argument('-t', action = 'store_true')
args = parser.parse_args()

test = args.t
num_time_periods = (2020 - 1988) * 12

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

def generate_compare_npy(year1, month1, num_months = 1):
    def inner_func(arr):
        year2, month2 = arr
        n_m = num_months_between(month2, year2, month1, year1)
        return n_m > 0 and n_m <= num_months
    return inner_func

def year_month_indices(yr_mth, yr_mth_idx, yr_mth_cts, year1, month1, num_months = 1):
    fn = generate_compare_npy(year1, month1, num_months)
    sel = np.apply_along_axis(fn, 1, yr_mth)

    idx_of_sel = sel.nonzero()[0]
    if len(idx_of_sel) == 0:
        return []
    start = yr_mth_idx[idx_of_sel[0]]
    end = yr_mth_idx[idx_of_sel[-1]] + yr_mth_cts[idx_of_sel[-1]]
    return list(range(start, end))

def calculate_avg_d2v(d2v1, d2v2, overlap=0, same=False):
    num_comp = d2v1.shape[0] * d2v2.shape[0] - overlap
    if num_comp <= 0:
        return np.nan, np.nan
    elif d2v1.shape[0] > 0 and d2v2.shape[0] > 0:
        cos_res = cosine_similarity(d2v1, d2v2)
        sum_d2v = np.sum(cos_res) - overlap
        num_comp = d2v1.shape[0] * d2v2.shape[0] - overlap
        avg_d2v = sum_d2v / num_comp
        if same:
            num_comp /= 2 # you're comparing to itself so 2way symmetry
        return avg_d2v, num_comp

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
            column index within list_idx2. This is normalized average cosine similarity,
            so (cos_sim + 1) / 2
        num_comp = number of comparisons made
    """
    num_comp = len(list_idx1) * len(list_idx2) - overlap
    if num_comp <= 0:
        return np.nan, np.nan
    else:
        list_idx1, list_idx2 = np.array(list_idx1), np.array(list_idx2)
        sum_d2v = np.sum(cos_res[list_idx1[:, None], list_idx2]) - overlap
        avg_d2v = sum_d2v / num_comp
        if same:
            num_comp /= 2
        return (avg_d2v + 1) / 2, num_comp
        
def generate_d2v_row(same_d2v, other_d2v, all_d2v, cos_res, col_info, replace=True):
    abrev_col, col, col_type1, col_type2, col_type3 = col_info
    d2v_dict = {}
    # same to same tracon
    avg_d2v, num_comp = calculate_avg_comp2(same_d2v, same_d2v, cos_res, \
            overlap = len(same_d2v), same = True)
    d2v_dict[f'trcn{col_type1}'] = avg_d2v
    d2v_dict[f'trcn{col_type2}'] = num_comp
    d2v_dict[f'trcn{col_type3}'] = len(same_d2v)

    # same to other tracon
    avg_d2v, num_comp = calculate_avg_comp2(same_d2v, other_d2v, cos_res)
    d2v_dict[f'trcn_invout{col_type1}'] = avg_d2v
    d2v_dict[f'trcn_invout{col_type2}'] = num_comp

    # other to other tracon
    avg_d2v, num_comp = calculate_avg_comp2(other_d2v, other_d2v, cos_res, \
            overlap = len(other_d2v), same = True)
    d2v_dict[f'trcn_out{col_type1}'] = avg_d2v
    d2v_dict[f'trcn_out{col_type2}'] = num_comp
    d2v_dict[f'trcn_out{col_type3}'] = len(other_d2v)

    # same to all tracon
    avg_d2v, num_comp = calculate_avg_comp2(same_d2v, all_d2v, cos_res,\
            overlap = len(same_d2v))
    d2v_dict[f'trcn_invall{col_type1}'] = avg_d2v
    d2v_dict[f'trcn_invall{col_type2}'] = num_comp

    # all to all tracon
    avg_d2v, num_comp = calculate_avg_comp2(all_d2v, all_d2v, cos_res, \
            overlap = len(all_d2v), same = True)
    d2v_dict[f'trcn_all{col_type1}'] = avg_d2v
    d2v_dict[f'trcn_all{col_type2}'] = num_comp
    d2v_dict[f'trcn_all{col_type3}'] = len(all_d2v)

    # b/w report1 and report2
    if col == 'narrative' or col == 'callback': # only those with mult reports
        this_tracon = searched.iloc[same_tracon, :]
        d2v_dict[f'trcn_mult_{abrev_col}{"_flfrm" if replace else ""}'] = \
                this_tracon[f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}'].mean().iloc[0]
        d2v_dict[f'trcn_mult_{abrev_col}{"_flfrm" if replace else ""}_ct'] = \
                this_tracon[f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}'].count().iloc[0]

    # return pd.Series(d2v_dict)
    return d2v_dict

abrev_col_dict = {'narrative': 'narr', 'synopsis': 'syn', \
        'narrative_synopsis_combined': 'narrsyn', 'combined': 'all', \
        'callback': 'call'}


def analyze_d2v(all_pds, d2v_model, replace = True, month_range_dict = {}, col = "", field_dict = {}, \
        tracon_month_unique=None, replace_dict={}):
    """
    @param: all_pds(pd.DataFrame) should be the full asrs dataset
    @param: d2v_model (gensim.models.doc2vec) model that was trained on full dataset
    @param: replace (bool): if true, the abbreviations were replaced by fullforms
    @param: month_range_dict (dict): month_range (1/3/6/12/inf) -> list of dataframes
        where each dataframe has the relevant doc2vec comparison info
    """
    if col == 'narrative' or col == 'callback': # only those with mult reports
        # mult_rep_cols = [f'{col}_report1',f'{col}_report2', \
        #         f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}',
        #         f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}']
        mult_rep_cols = []
    else:
        mult_rep_cols = []

    # all tracon
    all_trcn_codes = set(tracon_month_unique['tracon_code'])

    all_pds = all_pds[['tracon_code', 'year', 'month', col] + mult_rep_cols]
    all_pds.sort_values(['year', 'month', 'tracon_code'], inplace = True)

    yr_mth, yr_mth_idx, yr_mth_ct = np.unique(all_pds.values[:, [1, 2]].astype(int), \
            axis = 0, return_index=True, return_counts=True)

    abrev_col = abrev_col_dict[col]
    # for month_range in [1, 3, 6, 12, np.inf]:
    for month_range in [1, 3, 6, 12]:
        mr_str = f'{month_range}m'
        if month_range == np.inf:
            mr_str = 'atime'

        # used in d2v column names
        end_str = f"{abrev_col}_{mr_str}" 
        col_type1 = f'{"_flfrm" if replace else ""}_{end_str}'
        col_type2 = f'{"_flfrm" if replace else ""}_ct_{end_str}'
        col_type3 = f'{"_flfrm" if replace else ""}_vol_{end_str}'

        col_info = [col, abrev_col, col_type1, col_type2, col_type3]
        output_cols = [f'trcn{col_type1}', f'trcn{col_type2}', f'trcn{col_type3}', \
                f'trcn_invout{col_type1}', f'trcn_invout{col_type2}', \
                f'trcn_out{col_type1}', f'trcn_out{col_type2}', f'trcn_out{col_type3}', \
                f'trcn_invall{col_type1}', f'trcn_invall{col_type2}', \
                f'trcn_all{col_type1}', f'trcn_all{col_type2}', f'trcn_all{col_type3}']
        output_cols = {x: np.nan for x in output_cols}

        index_to_d2v = {}
        # generate dictionaries for caching
        tracon_month_dict, cos_res_tracon_dict = {}, {}
        start_time = time.time()
        tot_startup = 0
        tot_cos_sim = 0
        for month, year in tqdm(product(range(1, 13), range(1988, 2020)), total=num_time_periods, \
                desc = f'{col} {mr_str}'):
            startup_time = time.time()
            code = ' '.join([str(int(month)), str(int(year))])

            # select only the rows within the month range
            yr_mth_sel_idx = year_month_indices(yr_mth, yr_mth_idx, yr_mth_ct, int(year), int(month), \
                    num_months=month_range)

            tracon_month_dict[code] = all_pds.iloc[yr_mth_sel_idx, :].copy().drop_duplicates(col)
            searched = tracon_month_dict[code]

            total_range = np.array([True] * searched.shape[0])
            codes, code_idx, code_cts = np.unique(tracon_month_dict[code].values.astype(str)[:, 0], \
                    return_index=True, return_counts=True)

            sel = [x in all_trcn_codes for x in codes]
            codes, code_idx, code_cts = codes[sel], code_idx[sel], code_cts[sel]

            for code, c_idx, c_ct in zip(codes, code_idx, code_cts):
                total_range[c_idx:c_idx+c_ct] = False

            num_codes = codes.shape[0]
            missing_trcns = list(all_trcn_codes - set(codes))

            sum_comp, num_comp = np.zeros((num_codes + 1, num_codes + 1)), \
                    np.zeros((num_codes + 1, num_codes + 1)) 

            tot_startup += time.time() - startup_time
            if np.any(total_range):
                nontop50_idx = np.array([field_dict[str(x)] for x in searched.loc[total_range, col]])
                len_nontop50 = nontop50_idx.shape[0]
                # nontop50 = np.vstack([d2v_model.docvecs[field_dict[x]] for x in \
                #             searched.loc[total_range, col]])
            cos_start = time.time()
            for code_idx_i in range(num_codes):
                trcn1_idx = code_idx[code_idx_i], code_idx[code_idx_i] + code_cts[code_idx_i]
                processed_field = process_with_replace(x, replace_dict)
                trcn1 = np.vstack([d2v_model.docvecs[field_dict[processed_field]] for x in \
                        searched.iloc[trcn1_idx[0]:trcn1_idx[1]][col]])

                for code_idx_j in range(num_codes):
                    if code_idx_i <= code_idx_j:
                        trcn2_idx = code_idx[code_idx_j], code_idx[code_idx_j] + code_cts[code_idx_j]
                        processed_field = process_with_replace(x, replace_dict)
                        trcn2 = np.vstack([d2v_model.docvecs[field_dict[processed_field]] for x in \
                                searched.iloc[trcn2_idx[0]:trcn2_idx[1]][col]])

                        cos_res = cosine_similarity(trcn1, trcn2)

                        total_comp = code_cts[code_idx_i] * code_cts[code_idx_j]
                        if code_idx_i == code_idx_j:
                            total_comp = total_comp + code_cts[code_idx_i]
                        num_comp[code_idx_i, code_idx_j] = total_comp
                        sum_comp[code_idx_i, code_idx_j] = np.sum(cos_res)
                        if code_idx_i == code_idx_j:
                            sum_comp[code_idx_i, code_idx_j] += code_cts[code_idx_i]

                        num_comp[code_idx_j, code_idx_i] = num_comp[code_idx_i, code_idx_j]
                        sum_comp[code_idx_j, code_idx_i] = sum_comp[code_idx_i, code_idx_j]

                total = 0
                for idx in range(int(np.ceil(len_nontop50 / 500))):
                    processed_field = process_with_replace(x, replace_dict)
                    nontop50 = np.vstack([d2v_model.docvecs[field_dict[processed_field]] \
                            for x in nontop50_idx[idx*500: \
                            (idx+1)*500]])
                    total += np.sum(cosine_similarity(trcn1, nontop50))

                total_comp = trcn1.shape[0] * len_nontop50
                num_comp[code_idx_i, num_codes] = total_comp
                sum_comp[code_idx_i, num_codes] = total

                num_comp[num_codes, code_idx_i] = num_comp[code_idx_i, num_codes]
                sum_comp[num_codes, code_idx_i] = sum_comp[code_idx_i, num_codes]

            if np.any(total_range):
                total = 0
                for i in range(int(np.ceil(len_nontop50 / 2000))):
                    for j in range(int(np.ceil(len_nontop50 / 2000))):
                        processed_field = process_with_replace(x, replace_dict)
                        mat1 = np.vstack([d2v_model.docvecs[field_dict[processed_field]] for \
                                x in nontop50_idx[i*500:(i+1)*500]])
                        mat2 = np.vstack([d2v_model.docvecs[field_dict[processed_field]] for \
                                x in nontop50_idx[j*500:(j+1)*500]])
                        total += np.sum(cosine_similarity(mat1, mat2))
                total_comp = len_nontop50 ** 2 + len_nontop50
                num_comp[num_codes, num_codes] = total_comp
                sum_comp[num_codes, num_codes] = total + len_nontop50

            trcn_comp = np.sum(num_comp, axis=1)
            trcn_sum = np.sum(sum_comp, axis=1)

            diag_num = np.sum(np.diag(num_comp))
            diag_sum = np.sum(np.diag(sum_comp))

            all_num = np.sum(trcn_comp) / 2
            all_sum = np.sum(trcn_sum) / 2

            for idx, tracon in enumerate(codes):
                d2v_dict = {}
                comp = (num_comp[idx, idx] / 2 - code_cts[idx])
                if comp > 0:
                    d2v_dict[f'trcn{col_type1}'] = (1 + (sum_comp[idx, idx] / 2 - code_cts[idx]) / comp) / 2
                    d2v_dict[f'trcn{col_type2}'] = comp / 2
                    d2v_dict[f'trcn{col_type3}'] = code_cts[idx]
                else:
                    d2v_dict[f'trcn{col_type1}'] = np.nan
                    d2v_dict[f'trcn{col_type2}'] = np.nan
                    d2v_dict[f'trcn{col_type3}'] = np.nan

                # same to other tracon
                if trcn_comp[idx] - num_comp[idx, idx] > 0:
                    d2v_dict[f'trcn_invout{col_type1}'] = (1 + (trcn_sum[idx] - sum_comp[idx, idx]) / \
                            (trcn_comp[idx] - num_comp[idx, idx])) / 2
                    d2v_dict[f'trcn_invout{col_type2}'] = trcn_comp[idx] - num_comp[idx, idx]
                else:
                    d2v_dict[f'trcn_invout{col_type1}'] = np.nan
                    d2v_dict[f'trcn_invout{col_type2}'] = np.nan

                # other to other tracon
                num_other = searched.shape[0] - code_cts[idx]
                comp = (all_num - 2 * trcn_comp[idx] + num_comp[idx, idx]) / 2 - num_other
                if comp > 0:
                    tot = (all_sum - 2 * trcn_sum[idx] + sum_comp[idx, idx]) / 2 - num_other
                    d2v_dict[f'trcn_out{col_type1}'] = (1 + tot / comp) / 2
                    d2v_dict[f'trcn_out{col_type2}'] = comp
                    d2v_dict[f'trcn_out{col_type3}'] = searched.shape[0] - code_cts[idx]
                else:
                    d2v_dict[f'trcn_out{col_type1}'] = np.nan
                    d2v_dict[f'trcn_out{col_type2}'] = np.nan
                    d2v_dict[f'trcn_out{col_type3}'] = np.nan

                # same to all tracon
                comp = trcn_comp[idx] - code_cts[idx]
                if comp > 0:
                    d2v_dict[f'trcn_invall{col_type1}'] = (1 + (trcn_sum[idx] - code_cts[idx]) / comp) / 2
                    d2v_dict[f'trcn_invall{col_type2}'] = comp
                else:
                    d2v_dict[f'trcn_invall{col_type1}'] = np.nan
                    d2v_dict[f'trcn_invall{col_type2}'] = np.nan

                # all to all tracon
                if all_num > 0:
                    num_div = all_num - searched.shape[0]
                    d2v_dict[f'trcn_all{col_type1}'] = (1 + (all_sum - searched.shape[0]) / num_div) / 2
                    d2v_dict[f'trcn_all{col_type2}'] = num_div
                    d2v_dict[f'trcn_all{col_type3}'] = searched.shape[0]
                else:
                    d2v_dict[f'trcn_all{col_type1}'] = np.nan
                    d2v_dict[f'trcn_all{col_type2}'] = np.nan
                    d2v_dict[f'trcn_all{col_type3}'] = np.nan

                index_id = f"{tracon} {int(year)}/{int(month)}"
                index_to_d2v[index_id] = pd.Series(d2v_dict)

            row = output_cols.copy()
            num_div = all_num - searched.shape[0]
            if num_div > 0:
                row[f'trcn_all{col_type1}'] = (1 + (all_sum - searched.shape[0]) / num_div) / 2
                row[f'trcn_all{col_type2}'] = num_div
                row[f'trcn_all{col_type3}'] = searched.shape[0]

                row[f'trcn_out{col_type1}'] = (1 + (all_sum - searched.shape[0]) / num_div) / 2
                row[f'trcn_out{col_type2}'] = num_div
                row[f'trcn_out{col_type3}'] = searched.shape[0]

            mis_row = pd.Series(row)
            for tracon in missing_trcns:
                index_id = f"{tracon} {int(year)}/{int(month)}"
                index_to_d2v[index_id] = mis_row

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
        for dict_name in ['nasa', 'faa', 'casa', 'hand', 'iata']:
            dict_fullform = row[f'{dict_name}_fullform']
            if not pd.isna(dict_fullform):
                dict_fullform = str(dict_fullform).lower()
                replace_dict[row['acronym']] = dict_fullform
                break
    return replace_dict

def incident_unique_codes():
    # TODO: check that this pickle file is automatically being generated in pipeline
    unique_code_fn = '../results/unique_airport_code_ntsb_faa.pckl'
    unique_ntsb_faa_codes = pickle.load(open(unique_code_fn, 'rb'))
    # unique codes
    # return set(unique_ntsb_faa_codes)
    return unique_ntsb_faa_codes

def all_unique_codes(all_pds, tracon_month_unique, unique_codes):
    set_unique_codes = set(unique_codes)

    asrs_added_tracons = []
    for tracon_code in tracon_month_unique['tracon_code'].unique():
        if tracon_code not in set_unique_codes:
            asrs_added_tracons.append(tracon_code)

    # unique_ntsb_faa_codes
    return np.hstack([unique_codes, np.array(asrs_added_tracons)])

def filter_top50(unique_ntsb_faa_codes, tracon_month_unique):
    # if test:
    # if we're testing only utilize the airport codes from this wikipedia file
    top_50_iata = set(pd.read_excel('../datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    unique_ntsb_faa_codes = np.apply_along_axis(lambda x: [elem for elem in x if elem in top_50_iata], \
            0, unique_ntsb_faa_codes)
    tracon_month_unique = tracon_month_unique.loc[\
        tracon_month_unique['tracon_code'].apply(lambda x: x in top_50_iata)]
    return unique_ntsb_faa_codes, tracon_month_unique

def add_missing_rows(unique_ntsb_faa_codes, tracon_month_unique):
    all_combs = set(tracon_month_unique.apply(lambda x: (x[0], x[1], x[2]), axis = 1))
    added_rows = {'tracon_code': [], 'month': [], 'year':[]}
    for tracon, month, year in product(unique_ntsb_faa_codes, range(1, 13), range(1988, 2020)): 
        if (tracon, month, year) not in all_combs:
            added_rows['tracon_code'].append(tracon)
            added_rows['month'].append(month)
            added_rows['year'].append(year)

    return tracon_month_unique.append(pd.DataFrame.from_dict(added_rows))

def process_with_replace(field, r_d):
    np_res = replace_words(str(field), replace_dict=r_d)
    return ' '.join(np_res)

def generate_tagged_docs(np_fields, r_d):
    # creating list of tagged documents
    docs = []
    doc_to_idx = {}
    ct = 0
    for field in tqdm(np_fields, total=np_fields.shape[0]):
        doc_str = process_with_replace(field, r_d)
        if doc_str not in doc_to_idx:
            doc_to_idx[doc_str] = ct
            docs.append(TaggedDocument(doc_str, [ct]))
            ct += 1
    return docs, doc_to_idx
    
def d2v_multiple_reports(all_pds):
    for mult_col in ['narrative', 'callback']:
        reps = np.hstack((all_pds[f'{mult_col}_report1'].unique(), all_pds[f'{mult_col}_report2'].unique()))
        reps = reps.astype(str)

        for r_d in [load_replace_dictionary(mult_col), {}]:
            replace = len(r_d) > 0
            cos_col_name = f'{mult_col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}'
            all_pds[cos_col_name] = np.nan

            docs, doc_to_idx = generate_tagged_docs(reps, r_d)

            # train doc2vec
            model = Doc2Vec(docs, vector_size = 20, window = 3)
            only_mult_rep_df = all_pds.loc[all_pds[f'{mult_col}_multiple_reports'], :]
            for idx, row in tqdm(only_mult_rep_df.iterrows(), total=only_mult_rep_df.shape[0]):
                report1 = process_with_replace(row[f'{mult_col}_report1'], r_d)
                report2 = process_with_replace(row[f'{mult_col}_report2'], r_d)

                vec1 = model.docvecs[doc_to_idx[report1]]
                vec2 = model.docvecs[doc_to_idx[report2]]

                cos_sim = cosine_similarity(vec1.reshape(1, 20), vec2.reshape(1, 20))
                all_pds.loc[idx, cos_col_name] = (cos_sim[0, 0] + 1) / 2
    return all_pds

def cos_sim_analysis(all_pds, tracon_month_unique):
    # for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
    for col in ['callback', 'combined', 'narrative_synopsis_combined']:
        month_range_dict = {}
        for r_d in [load_replace_dictionary(col), {}]:
            reps = all_pds[col].unique()
            docs, doc_to_idx = generate_tagged_docs(reps, r_d)
            # train doc2vec
            print('training doc2vec models. This can take a while...')
            model = Doc2Vec(docs, vector_size = 20, window = 3)

            analyze_d2v(all_pds, model, len(r_d) > 0, month_range_dict, col = col, field_dict = doc_to_idx, \
                    tracon_month_unique=tracon_month_unique)

        for month_range in month_range_dict.keys():
            res = pd.concat(month_range_dict[month_range], axis = 1)
            res.to_csv(f'results/d2v_tracon_month_{col}_{month_range}mon.csv')

def main():
    # load files
    all_pds = load_asrs(load_saved = True)
    all_pds = all_pds.reset_index().drop('index', axis = 1)
    # all_pds = tracon_analysis(all_pds)

    # top 50/missing row analysis
    tracon_month_unique = all_pds[['tracon_code', 'month', 'year']].drop_duplicates()
    unique_codes = incident_unique_codes()
    unique_ntsb_faa_codes = all_unique_codes(all_pds, tracon_month_unique, unique_codes)
    print('after missing row analysis')

    unique_ntsb_faa_codes, tracon_month_unique = filter_top50(unique_ntsb_faa_codes, tracon_month_unique)
    tracon_month_unique = add_missing_rows(unique_ntsb_faa_codes, tracon_month_unique)
    print('after missing rows')

    all_pds = d2v_multiple_reports(all_pds)
    cos_sim_analysis(all_pds, tracon_month_unique)

if __name__ == "__main__":
    main()
