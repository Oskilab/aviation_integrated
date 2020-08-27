import pandas as pd, numpy as np, re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython import embed
from collections import Counter
from tqdm import tqdm
from preprocess_helper import *
from sklearn.metrics.pairwise import cosine_similarity

all_pds = load_asrs(load_saved = True)

def num_months_between(month1, year1, month2, year2):
    return (year2 - year1) * 12 + month2 - month1

def generate_compare(month1, year1, num_months = 1): # accident date
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
    num_comp = len(list_idx1) * len(list_idx2) - overlap
    if num_comp == 0:
        return np.nan, np.nan
    else:
        list_idx1, list_idx2 = np.array(list_idx1), np.array(list_idx2)
        sum_d2v = cos_res[list_idx1[:, None], list_idx2]
        avg_d2v = sum_d2v / num_comp
        if same:
            num_comp /= 2
        return avg_d2v, num_comp
        

"""
@param: all_pds(pd.DataFrame) should be the full asrs dataset
@param: d2v_model (gensim.models.doc2vec) model that was trained on full dataset
@param: replace (bool): if true, the abbreviations were replaced by fullforms
@param: month_range_dict (dict): month_range (1/3/6/12/inf) -> list of dataframes
    where each dataframe has the relevant doc2vec comparison info
"""
def analyze_d2v(all_pds, d2v_model, replace = True, month_range_dict = {}, col = ""):
    for month_range in [1, 3, 6, 12, np.inf]:
        tracon_month_dict, tracon_all_dict = {}, {}
        cos_res_tracon_dict = {}
        month_year_pd = all_pds[['month', 'year']].drop_duplicates()
        for idx, date_row in tqdm(month_year_pd.iterrows(), total = month_year_pd.shape[0], desc = \
                f"{col} dict creation {month_range}mon"):
            code = ' '.join([str(date_row['month']), str(date_row['year'])])
            compare_func = generate_compare(date_row['month'], date_row['year'], num_months = month_range)
            tracon_month_dict[code] = all_pds.loc[all_pds.apply(compare_func, axis = 1), :].copy()

            all_tracon = list(tracon_month_dict[code].drop_duplicates(col).index)
            # avg_d2v, num_comp = calculate_avg_comp(all_tracon, all_tracon, d2v_model, \
            #         overlap = len(all_tracon), same = True)
            # tracon_all_dict[code] = (avg_d2v, num_comp)

            if len(all_tracon) >= 1:
                d2v1 = np.vstack([d2v_model.docvecs[x] for x in all_tracon])
                cos_res = cosine_similarity(d2v1, d2v1)
                cos_res_tracon_dict[code] = cos_res

        index_to_d2v  = {}
        tracon_month_unique = all_pds[['tracon_code', 'month', 'year']].drop_duplicates()

        for idx, row in tqdm(tracon_month_unique.iterrows(), total = tracon_month_unique.shape[0], \
                desc = f"{col} analysis {month_range}mon"):
            index_id = f"{row['tracon_code']} {row['year']}/{row['month']}"
            code = ' '.join([str(row['month']), str(row['year'])])

            if code in cos_res_tracon_dict:
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
                # avg_d2v, num_comp = calculate_avg_comp(same_tracon, same_tracon, d2v_model,\
                #         overlap = len(same_tracon), same = True)
                avg_d2v, num_comp = calculate_avg_comp2(same_tracon, same_tracon, cos_res, \
                        overlap = len(same_tracon), same = True)
                d2v_dict[f'd2v_within_tracon{"_replace" if replace else ""}'] = avg_d2v
                d2v_dict[f'd2v_num_comp_within_tracon{"_replace" if replace else ""}'] = num_comp

                # same to other tracon
                # avg_d2v, num_comp = calculate_avg_comp(same_tracon, other_tracon, d2v_model)
                avg_d2v, num_comp = calculate_avg_comp2(same_tracon, other_tracon, cos_res)
                d2v_dict[f'd2v_tracon_to_other{"_replace" if replace else ""}'] = avg_d2v
                d2v_dict[f'd2v_num_comp_tracon_to_other{"_replace" if replace else ""}'] = num_comp

                # other to other tracon
                # avg_d2v, num_comp = calculate_avg_comp(other_tracon, other_tracon, d2v_model, \
                #         overlap = len(other_tracon), same = True)
                avg_d2v, num_comp = calculate_avg_comp2(other_tracon, other_tracon, cos_res, \
                        overlap = len(other_tracon), same = True)
                d2v_dict[f'd2v_other_to_other{"_replace" if replace else ""}'] = avg_d2v
                d2v_dict[f'd2v_num_comp_other_to_other{"_replace" if replace else ""}'] = num_comp

                # same to all tracon
                # avg_d2v, num_comp = calculate_avg_comp(same_tracon, all_tracon, d2v_model, \
                #         overlap = len(same_tracon))
                avg_d2v, num_comp = calculate_avg_comp2(same_tracon, all_tracon, cos_res, \
                        overlap = len(same_tracon))
                d2v_dict[f'd2v_tracon_to_all{"_replace" if replace else ""}'] = avg_d2v
                d2v_dict[f'd2v_num_comp_tracon_to_all{"_replace" if replace else ""}'] = num_comp

                # all to all tracon
                # avg_d2v, num_comp = calculate_avg_comp(all_tracon, all_tracon, d2v_model, \
                #         overlap = len(all_tracon), same = True)
                avg_d2v, num_comp = calculate_avg_comp2(all_tracon, all_tracon, cos_res, \
                        overlap = len(all_tracon), same = True)
                d2v_dict[f'd2v_all_to_all{"_replace" if replace else ""}'] = tracon_all_dict[code][0]
                d2v_dict[f'd2v_num_comp_all_to_all{"_replace" if replace else ""}'] = \
                        tracon_all_dict[code][1]

                # # b/w report1 and report2
                if col == 'narrative' or col == 'callback': # only those with mult reports
                    this_tracon = searched.loc[searched['tracon_code'] == row['tracon_code'], :].copy()
                    this_tracon.drop_duplicates([f'{col}_report1', f'{col}_report2'], inplace = True)
                    d2v_dict[f'd2v_avg_between_reports{"_replace" if replace else ""}'] = \
                            this_tracon[f'{col}_multiple_reports_cos_sim{"_replace" if replace else ""}'].mean()
                    d2v_dict[f'd2v_num_between_reports{"_replace" if replace else ""}'] = \
                            this_tracon[f'{col}_multiple_reports_cos_sim{"_replace" if replace else ""}'].count()

                index_to_d2v[index_id] = pd.Series(d2v_dict)
        fin = pd.DataFrame.from_dict(index_to_d2v, orient = 'index')
        month_range_dict[month_range] = month_range_dict.get(month_range, []) + [fin]

def load_replace_dictionary(col):
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

dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'iata']
all_pds = all_pds.reset_index().drop('index', axis = 1)

# deal with multiple reports
# for mult_col in ['narrative', 'callback']:
#     for r_d in [load_replace_dictionary(mult_col), {}]:
#         replace = len(r_d) > 0
#         cos_col_name = f'{mult_col}_multiple_reports_cos_sim{"_replace" if replace else ""}'
#         all_pds[cos_col_name] = np.nan
#
#         # creating list of tagged documents
#         docs, set_of_docs = [], set()
#         for idx, row in tqdm(all_pds.iterrows(), total = all_pds.shape[0], desc = 'creating' + \
#                 f' documents for {mult_col}{" replace" if replace else ""}'):
#             field1 = row[f'{mult_col}_report1']
#             if field1 not in set_of_docs and not pd.isna(field1):
#                 dup_idx = all_pds.loc[all_pds[f'{mult_col}_report1'] == field1, :].index
#                 doc_str = ' '.join(convert_to_words(row, f'{mult_col}_report1', r_d))
#                 docs.append(TaggedDocument(doc_str, [f'{index} 1' for index in dup_idx]))
#                 set_of_docs.add(field1)
#
#             field2 = row[f'{mult_col}_report2']
#             if field2 not in set_of_docs and not pd.isna(field2):
#                 dup_idx = all_pds.loc[all_pds[f'{mult_col}_report2'] == field2, :].index
#                 doc_str = ' '.join(convert_to_words(row, f'{mult_col}_report2', r_d))
#                 docs.append(TaggedDocument(doc_str, [f'{index} 2' for index in dup_idx]))
#                 set_of_docs.add(field2)
#
#         model = Doc2Vec(docs, vector_size = 20, window = 3)
#         only_mult_rep_df = all_pds.loc[all_pds[f'{mult_col}_multiple_reports'], :]
#         for idx, row in only_mult_rep_df.iterrows():
#             vec1, vec2 = model.docvecs[f'{idx} 1'], model.docvecs[f'{idx} 2']
#             cos_sim = cosine_similarity(vec1.reshape(1, 20), vec2.reshape(1, 20))
#             all_pds.loc[idx, cos_col_name] = cos_sim[0, 0]
#

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
