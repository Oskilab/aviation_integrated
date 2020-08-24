import pandas as pd, numpy as np, re
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from IPython import embed
from tqdm import tqdm
from preprocess_helper import *
from sklearn.metrics.pairwise import cosine_similarity

all_pds = load_asrs()

def num_months_between(month1, year1, month2, year2):
    return (year2 - year1) * 12 + month2 - month1

def generate_compare(month1, year1, num_months = 1): # accident date
    def inner_func(row):
        month2, year2 = row['month'], row['year']
        n_m = num_months_between(month1, year1, month2, year2)
        return n_m > 0 and n_m <= num_months
    return inner_func

"""
@param: all_pds(pd.DataFrame) should be the full asrs dataset
@param: d2v_model (gensim.models.doc2vec) model that was trained on full dataset
@param: replace (bool): if true, the abbreviations were replaced by fullforms
@param: month_range_dict (dict): month_range (1/3/6/12/inf) -> list of dataframes
    where each dataframe has the relevant doc2vec comparison info
"""
def analyze_d2v(all_pds, d2v_model, replace = True, month_range_dict = {}):
    for month_range in [1, 3, 6, 12, np.inf]:
        tracon_month_dict = {}
        print('first')
        for idx, date_row in tqdm(all_pds[['month', 'year']].drop_duplicates().iterrows()):
            code = ' '.join([str(date_row['month']), str(date_row['year'])])
            compare_func = generate_compare(date_row['month'], date_row['year'], num_months = month_range)
            tracon_month_dict[code] = all_pds.loc[all_pds.apply(compare_func, axis = 1), :].copy()

        index_to_d2v, index_to_d2v_other, index_to_d2v_all = {}, {}, {}
        print('second')
        for idx, row in tqdm(all_pds.iterrows()):
            index_id = f"{row['tracon_code']} {row['year']}/{row['month']}"
            code = ' '.join([str(row['month']), str(row['year'])])

            found_d2v = None

            # same tracon
            searched = tracon_month_dict[code]
            searched = searched.loc[searched['tracon_code'] == row['tracon_code'], :]

            if searched.shape[0] > 1:
                d2v_list = [d2v_model.docvecs[x] for x in list(searched.index)]
                d2v_sub = np.vstack(d2v_list)
                found_d2v = d2v_sub

                cos_res = cosine_similarity(d2v_sub, d2v_sub)
                sum_d2v = np.sum(cos_res) - np.sum(np.diagonal(cos_res))

                num_comp = cos_res.shape[0] * (cos_res.shape[0] - 1)
                avg_d2v = sum_d2v / num_comp
                num_comp /= num_comp
            else:
                avg_d2v = np.nan
                num_comp = np.nan
            index_to_d2v[index_id] = pd.Series({f'd2v_cos_sim{"_replace" if replace else ""}': avg_d2v,\
                    f'd2v_num_comp{"_replace" if replace else ""}': num_comp})

            # other tracon
            searched = tracon_month_dict[code]
            searched = searched.loc[searched['tracon_code'] != row['tracon_code'], :]
            if searched.shape[0] > 1 and found_d2v is not None:
                d2v_list = [d2v_model.docvecs[x] for x in list(searched.index)]
                d2v_sub = np.vstack(d2v_list)

                cos_res = cosine_similarity(found_d2v, d2v_sub)
                sum_d2v = np.sum(cos_res)
                num_comp = cos_res.shape[0] * cos_res.shape[1]
                avg_d2v = sum_d2v / num_comp
            else:
                avg_d2v = np.nan
                num_comp = np.nan
            index_to_d2v_other[index_id] = pd.Series({\
                    f'd2v_cos_sim_other_tracon{"_replace" if replace else ""}': avg_d2v, \
                    f'd2v_num_comp_other_tracon{"_replace" if replace else ""}': num_comp})

            # all tracons
            searched = tracon_month_dict[code]
            if searched.shape[0] > 1 and found_d2v is not None:
                d2v_list = [d2v_model.docvecs[x] for x in list(searched.index)]
                d2v_sub = np.vstack(d2v_list)

                cos_res = cosine_similarity(found_d2v, d2v_sub)
                sum_d2v = np.sum(cos_res) - cos_res.shape[0] # comparisons to itself
                num_comp = (cos_res.shape[0] * cos_res.shape[1] - cos_res.shape[0])
                avg_d2v = sum_d2v / num_comp
            else:
                avg_d2v = np.nan
                num_comp = np.nan
            index_to_d2v_all[index_id] = pd.Series({\
                    f'd2v_cos_sim_all_tracon{"_replace" if replace else ""}': avg_d2v, \
                    f'd2v_num_comp_all_tracon{"_replace" if replace else ""}': num_comp})
        same = pd.DataFrame.from_dict(index_to_d2v, orient = 'index')
        other = pd.DataFrame.from_dict(index_to_d2v_other, orient = 'index')
        all_tr = pd.DataFrame.from_dict(index_to_d2v_all, orient = 'index')
        fin = pd.concat([same, other, all_tr], axis = 1)
        month_range_dict[month_range] = month_range_dict.get(month_range, []) + [fin]


dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'iata']
all_pds = all_pds.reset_index().drop('index', axis = 1)
for col in ['narrative', 'synopsis', 'combined']:

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
    # train doc2vec
    print('training doc2vec models. This can take a while...')
    month_range_dict = {}
    for r_d in [replace_dict, {}]:
        docs = [TaggedDocument(' '.join(x), [idx]) for idx, x in \
                enumerate(all_pds.apply(lambda x: convert_to_words(x, col, r_d), axis = 1))]
        model = Doc2Vec(docs, vector_size = 20, window = 3)
        analyze_d2v(all_pds, model, len(r_d) > 0, month_range_dict)

    for month_range in month_range_dict.keys():
        res = pd.concat(month_range_dict[month_range], axis = 1)
        res.to_csv(f'results/d2v_tracon_month_{col}_{month_range}mon.csv')
