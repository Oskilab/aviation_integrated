"""
Calculates the average cosine similarity.
"""
import pickle
import argparse
from itertools import product

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from IPython import embed

import pandas as pd
import numpy as np
import preprocess_helper

parser = argparse.ArgumentParser(description='Calculate average cosine similarity.')
parser.add_argument('-t', action='store_true')
parser.add_argument('--lag', default=1, const=1, nargs='?', type=int)
args = parser.parse_args()

test = args.t
lag = args.lag
NUM_TIME_PERIODS = (2020 - 1988) * 12

def generate_d2v_vecs(pd_df, d2v_model, field_dict, replace_dict, use_field_dict=True):
    """
    This generates a matrix representing the d2v vectors for all the documents within
    the iterable pd_df.
    @param: pd_df (iterable of str documents)
    @param: d2v_model (gensim.models.Doc2Vec) converts document to vector
    @param: field_dict (dict[document] -> index) converts document to index for
        gensim.models.Doc2Vec model (each document is tagged with an index)
    @param: replace_dict (dict[orig_word] -> word) converts words to fullform version
        of the word. This depends on whether or not we are replacing abbrevations with
        their fullform
    @param: use_field_dict (bool) determines whether or not pd_df is an iterable of documents
        (occurs when True), or an iterable of index (of gensim.models.Doc2Vec model)
    @returns: np.ndarray with shape (len(pd_df), d2v dimension)
    """
    arr = []
    for field in pd_df:
        if use_field_dict:
            arr += [d2v_model.docvecs[field_dict[process_with_replace(field, replace_dict)]]]
        else:
            arr += [d2v_model.docvecs[field]]
    return np.vstack(arr)

def generate_nontop50_info(num_rows, codes_info):
    """
    This generates a np.ndarray of booleans, which indicate whether or not each index of the
    a given dataframe is covered by our top50 iata codes (currently we are only utilizing
    top 50 iata codes within our analysis, total_range[idx] = True if the idx row of our dataframe
    is not in the top50 iata codes).
    @param: num_rows (int) number of rows in our given dataframe
    @param: codes_info (list of np.ndarray)
        codes_info[0] = np.ndarray([code1, ..., coden]), an np.ndarray of unique codes in our df
        codes_info[1] = np.ndarray([code1_idx, ... coden_idx]), an np.ndarray of indices of each
            unique code in our dataframe. We assume the dataframe is sorted by codes
        codes_info[2] = np.ndarray([code1_ct, ..., coden_ct]) an np.ndarray of counts of each
            unique code in our dataframe.
        See np.unique documentation for details
    @returns: total_range (np.ndarray of booleans), where total_range[idx] = True if idx of df
        is in top50 iata codes set.
    """
    codes, code_idx, code_cts = codes_info

    total_range = np.array([True] * num_rows)
    for _, c_idx, c_ct in zip(codes, code_idx, code_cts):
        total_range[c_idx:c_idx+c_ct] = False
    return total_range

def generate_codes_info(searched, all_trcn_codes):
    """
    This returns unique code information (in format described below). Note that we only select
    the portion of the dataframe within the top50 iata codes.
    @param: searched (pd.DataFrame), the current dataframe. First column must be tracon_code
    @param: all_trcn_codes (set of codes), a set of the top50 iata codes
    @returns: codes_info for only the top50 iata codes (format is above)
    """
    codes, code_idx, code_cts = np.unique(searched.values.astype(str)[:, 0], \
            return_index=True, return_counts=True)
    sel = [x in all_trcn_codes for x in codes]
    return [codes[sel], code_idx[sel], code_cts[sel]]

def num_comparisons(code_cts, code_idx_i, code_idx_j):
    """
    If code_i and code_j are distinct, then we return the number of pairwise comparisons or
    ct_i * ct_j. However, if they are the same (meaning we are computing pairwise comparisons
    within one group/tracon), then we calculate the number of comparisons as ct_i * (ct_i + 1)
    @param: code_cts (np.ndarray), format described above.
    @param: code_idx_i (int) indexes into code_cts
    @param: code_idx_j (int) indexes into code_cts
    @returns: modified number of comparisons between two tracons (in a given dataframe)
    """
    num_comp = code_cts[code_idx_i] * code_cts[code_idx_j]
    if code_idx_i == code_idx_j:
        num_comp = num_comp + code_cts[code_idx_i]
    return num_comp

def sum_comparisons(cos_res, code_cts, code_idx_i, code_idx_j):
    """
    Calculates the total sum of comparisons between two tracon codes. If code_i and code_j are not
    distinct, then we calculate the sum as sum_ij + ct_i.
    @param: cos_res (np.ndarray) w/shape (code_cts[code_idx_i], code_cts[code_idx_j])
        this is a 2d np.ndarray where the rows index into a particular row with tracon code i
        and the columns index into a particular row with tracon code j
    @param: code_cts (np.ndarray), format described above.
    @param: code_idx_i (int) indexes into code_cts
    @param: code_idx_j (int) indexes into code_cts
    @returns: modified sum of comparisons between tracon_i and tracon_j.
    """
    sum_comp = np.sum(cos_res)
    if code_idx_i == code_idx_j:
        sum_comp += code_cts[code_idx_i]
    return sum_comp


def generate_trcn_vecs(searched, d2v_model, idx, code_idx, code_cts, field_dict, replace_dict, col):
    """
    Generates a matrix for the ith tracon_code filled with the doc2vec representations.
    @param: searched (pd.DataFrame) the dataframe we are analyzing
    @param: d2v_model (gensim.models.Doc2Vec) converts documents to doc2vec representations
    @param: idx (int) indexes into code_idx/code_cts, indicates which code we are looking
    @param: code_idx (np.ndarray) contains start_index (within searched df) of each code
    @param: code_cts (np.ndarray) contains number of times the code appears in searched
    @param: field_dict (dict[document] -> index) converts document to index for
        gensim.models.Doc2Vec model (each document is tagged with an index)
    @param: replace_dict (dict[orig_word] -> word) converts words to fullform version
        of the word. This depends on whether or not we are replacing abbrevations with
        their fullform
    @param: col (str) which column we are analyzing
    """
    trcn1_idx = code_idx[idx], code_idx[idx] + code_cts[idx]
    return generate_d2v_vecs(searched.iloc[trcn1_idx[0]:trcn1_idx[1]][col], \
            d2v_model, field_dict, replace_dict)

def populate_comp_matrix(searched, d2v_model, field_dict, codes_info, replace_dict, col):
    """
    This function calculates two matrices (num_comp and sum_comp). We are trying to calculate the
    average doc2vec cos_sim between any pair of tracon_months. To do so, we take any time period
    (e.g., Sep. 2011) and calculate two matrices each of size (#tracons + 1, #tracons + 1).
        num_comp[i, j] = # of (adjusted) pairwise comparisons between tracon i and
            j within a time period
        sum_comp[i, j] = sum of all pairwise comparisons between i/j within time period
    The last row/column keeps track of the same information for all tracons outside of the top
    50 iata codes (which we are limiting our analysis to).

    What's with the adjustment? notice that if you are comparing tracon_i to tracon_i, the number
    of comparisons is going to be different than if you compared two distinct tracons. Furthermore,
    we will calculate avg number by adding up cells in the final output, but doing so will double
    the off-diagonal terms (due to symmetry) and single-count the on-diagonal terms. To deal with
    this issue, we modify the on-diagonal terms to double-count those comparisons, and simply divide
    the sum by 2 (there may be a better way to do this, but this method was developed while
    optimizing performance).

    Using num_comp/sum_comp, we should be able to calculate all the d2v terms by adding the relevant
    terms together (for any given time period), and dividing by the number of comparisons.

    @param: searched (pd.DataFrame) a dataframe consisting of rows only from a one-month time period
        (e.g., Sep 2011)
    @param: d2v_model (gensim.models.Doc2Vec) converts documents to doc2vec representations
    @param: field_dict (dict[document] -> index) converts document to index for
        gensim.models.Doc2Vec model (each document is tagged with an index)
    @param: all_trcn_codes (set of codes), a set of the top50 iata codes
    @param: codes_info for only the top50 iata codes (format is above)
    @param: replace_dict (dict[orig_word] -> word) converts words to fullform version
        of the word. This depends on whether or not we are replacing abbrevations with
        their fullform
    @param: col (str) column we are analyzing
    @returns: num_comp/sum_comp (np.ndarray), described above
    """
    # preliminary info
    codes, code_idx, code_cts = codes_info
    total_range = generate_nontop50_info(searched.shape[0], codes_info)

    num_codes = codes.shape[0]

    sum_comp, num_comp = np.zeros((num_codes + 1, num_codes + 1)), \
            np.zeros((num_codes + 1, num_codes + 1))

    # nontop50 indices for gensim model
    if np.any(total_range):
        nontop50_idx = np.array([field_dict[process_with_replace(x, replace_dict)] for \
                x in searched.loc[total_range, col]])
        len_nontop50 = nontop50_idx.shape[0]

    # populations sum_comp/num_comp
    for code_idx_i in range(num_codes):
        # generate vectors for tracon i
        trcn1 = generate_trcn_vecs(searched, d2v_model, code_idx_i, code_idx, code_cts, \
                field_dict, replace_dict, col)

        for code_idx_j in range(num_codes):
            if code_idx_i <= code_idx_j:
                # generate vectors for tracon j
                trcn2 = generate_trcn_vecs(searched, d2v_model, code_idx_j, code_idx, code_cts, \
                        field_dict, replace_dict, col)

                # pairwise comparison matrix
                cos_res = cosine_similarity(trcn1, trcn2)

                total_comp = num_comparisons(code_cts, code_idx_i, code_idx_j)

                num_comp[code_idx_i, code_idx_j] = total_comp
                sum_comp[code_idx_i, code_idx_j] = sum_comparisons(cos_res, code_cts, \
                        code_idx_i, code_idx_j)

                # fill in symmetrically
                num_comp[code_idx_j, code_idx_i] = num_comp[code_idx_i, code_idx_j]
                sum_comp[code_idx_j, code_idx_i] = sum_comp[code_idx_i, code_idx_j]

        if np.any(total_range):
            # nontop50 comparisons x one top50 comparisons
            total = 0
            for idx in range(int(np.ceil(len_nontop50 / 500))):
                nontop50 = generate_d2v_vecs(nontop50_idx[idx*500:(idx+1)*500], \
                        d2v_model, field_dict, replace_dict, use_field_dict=False)
                total += np.sum(cosine_similarity(trcn1, nontop50))

            total_comp = trcn1.shape[0] * len_nontop50
            num_comp[code_idx_i, num_codes] = total_comp
            sum_comp[code_idx_i, num_codes] = total

            num_comp[num_codes, code_idx_i] = num_comp[code_idx_i, num_codes]
            sum_comp[num_codes, code_idx_i] = sum_comp[code_idx_i, num_codes]

    return num_comp, sum_comp

def analyze_time_period(searched, num_comp, sum_comp, code_info, col_types, default_dict, \
        index_to_d2v, year, month):
    """
    This analyzes a given time period (e.g., Sep. 2011) by calculating the average cos_sim numbers
    for each tracon within the given time period, and saves the results into pandas Series. This
    is then saved to a dictionary file (which is converted to a dataframe outside of this function).
    @param: searched (pd.DataFrame), the current dataframe. First column must be tracon_code
    @param: num_comp (np.ndarray), described above
    @param: sum_comp (np.ndarray), described above
    @param: code_info (list of np.ndarray)
        codes_info[0] = np.ndarray([code1, ..., coden]), an np.ndarray of unique codes in our df
        codes_info[1] = np.ndarray([code1_idx, ... coden_idx]), an np.ndarray of indices of each
            unique code in our dataframe. We assume the dataframe is sorted by codes
        codes_info[2] = np.ndarray([code1_ct, ..., coden_ct]) an np.ndarray of counts of each
            unique code in our dataframe.
        See np.unique documentation for details
    @param: col_types (list of column names), col_types[0] = d2v, col_types[1] = # pairwise
        comparisons, col_types[2] = volume (or # of rows of that tracon)
    @param: default_dict (dict[col] -> val) this is the default dictionary for the pd.Series we
    are generating
    @param: index_to_d2v (dict[tracon_month] -> pd.Series) this is the dictionary we are building up
        maps tracon_month to each row
    @returns: pd.Series for all tracon_codes that do not show up within this given time period
        (all of them have the same values for this time period).
    """
    codes, _, code_cts = code_info
    col_type1, col_type2, col_type3 = col_types

    # preliminary info
    trcn_comp = np.sum(num_comp, axis=1)
    trcn_sum = np.sum(sum_comp, axis=1)

    all_num = np.sum(trcn_comp) / 2
    all_sum = np.sum(trcn_sum) / 2

    for idx, tracon in enumerate(codes):
        d2v_dict = default_dict.copy()
        comp = (num_comp[idx, idx] / 2 - code_cts[idx])
        if comp > 0:
            d2v_dict[f'trcn{col_type1}'] = (1 + (sum_comp[idx, idx] / 2 - code_cts[idx]) / comp) / 2
            d2v_dict[f'trcn{col_type2}'] = comp
            d2v_dict[f'trcn{col_type3}'] = code_cts[idx]
        elif code_cts[idx] == 1:
            d2v_dict[f'trcn{col_type3}'] = code_cts[idx]

        # other to other tracon
        num_other = searched.shape[0] - code_cts[idx]
        comp = (all_num - 2 * trcn_comp[idx] + num_comp[idx, idx]) / 2 - num_other
        if comp > 0:
            tot = (all_sum - 2 * trcn_sum[idx] + sum_comp[idx, idx]) / 2 - num_other
            d2v_dict[f'trcn_out{col_type1}'] = (1 + tot / comp) / 2
            d2v_dict[f'trcn_out{col_type2}'] = comp
            d2v_dict[f'trcn_out{col_type3}'] = searched.shape[0] - code_cts[idx]
        elif searched.shape[0] - code_cts[idx] == 1:
            d2v_dict[f'trcn_out{col_type3}'] = searched.shape[0] - code_cts[idx]

        # same to all tracon
        comp = trcn_comp[idx] - code_cts[idx]
        if comp > 0:
            d2v_dict[f'trcn_invall{col_type1}'] = (1 + (trcn_sum[idx] - code_cts[idx]) / comp) / 2
            d2v_dict[f'trcn_invall{col_type2}'] = comp

        index_id = f"{tracon} {int(year)}/{int(month)}"
        index_to_d2v[index_id] = pd.Series(d2v_dict)

    row = default_dict.copy()
    num_div = all_num - searched.shape[0]
    if num_div > 0:
        row[f'trcn_out{col_type1}'] = (1 + (all_sum - searched.shape[0]) / num_div) / 2
        row[f'trcn_out{col_type2}'] = num_div
        row[f'trcn_out{col_type3}'] = searched.shape[0]

    return pd.Series(row)

def analyze_d2v(all_pds, d2v_model, replace=True, month_range_dict={}, col="", field_dict={}, \
        tracon_month_unique=None, replace_dict={}, lag=1):
    """
    Performs d2v cos_sim calculations for one particular column.
    @param: all_pds(pd.DataFrame) should be the full asrs dataset
    @param: d2v_model (gensim.models.doc2vec) model that was trained on full dataset
    @param: replace (bool): if true, the abbreviations were replaced by fullforms
    @param: month_range_dict (dict): month_range (1/3/6/12/inf) -> list of dataframes
        where each dataframe has the relevant doc2vec comparison info
    @param: col (str) the column we are analyzing
    @param: field_dict (dict[document] -> index) converts document to index for
        gensim.models.Doc2Vec model (each document is tagged with an index)
    @param: tracon_month_unique (pd.DataFrame) with only columns = tracon_code/year/month
        that lists out unique tracon_months within dataset
    @param: replace_dict (dict[orig_word] -> word) converts words to fullform version
        of the word. This depends on whether or not we are replacing abbrevations with
        their fullform
    """
    if col in ['narrative', 'callback']: # only those with mult reports
        # mult_rep_cols = [f'{col}_report1',f'{col}_report2', \
        #         f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}',
        #         f'{col}_multiple_reports_cos_sim{"_flfrm" if replace else ""}']
        mult_rep_cols = []
    else:
        mult_rep_cols = []

    # all tracon
    all_trcn_codes = set(tracon_month_unique['tracon_code'])

    all_pds = all_pds[['tracon_code', 'year', 'month', col] + mult_rep_cols]
    all_pds.sort_values(['year', 'month', 'tracon_code'], inplace=True)

    yr_mth, yr_mth_idx, yr_mth_ct = np.unique(all_pds.values[:, [1, 2]].astype(int), \
            axis=0, return_index=True, return_counts=True)

    abrev_col = preprocess_helper.ABREV_COL_DICT[col]
    for month_range in [1, 3, 6, 12]:
        mr_str = f'{month_range}m'
        if month_range == np.inf:
            mr_str = 'atime'

        # used in d2v column names
        end_str = f"{abrev_col}_{mr_str}"
        col_type1 = f'{"_ff" if replace else ""}_{end_str}'
        col_type2 = f'{"_ff" if replace else ""}_ct_{end_str}'
        col_type3 = f'{"_ff" if replace else ""}_vol_{end_str}'
        col_types = [col_type1, col_type2, col_type3]

        output_cols = [f'trcn{col_type1}', f'trcn{col_type2}', f'trcn{col_type3}', \
                f'trcn_out{col_type1}', f'trcn_out{col_type2}', f'trcn_out{col_type3}', \
                f'trcn_invall{col_type1}', f'trcn_invall{col_type2}']
        output_cols = {x: np.nan for x in output_cols}

        index_to_d2v = {}
        for month, year in tqdm(product(range(1, 13), range(1988, 2020)), total=NUM_TIME_PERIODS, \
                desc=f'{col} {mr_str}'):
            # select only the rows within the month range
            yr_mth_sel_idx = preprocess_helper.year_month_indices(yr_mth, yr_mth_idx, yr_mth_ct, \
                    int(year), int(month), num_months=month_range, lag=lag)

            # drop duplicates of given column
            searched = all_pds.iloc[yr_mth_sel_idx, :].copy()

            codes_info = generate_codes_info(searched, all_trcn_codes)
            num_comp, sum_comp = populate_comp_matrix(searched, d2v_model, \
                    field_dict, codes_info, replace_dict, col)

            # adds all series to index_to_d2v
            mis_row = analyze_time_period(searched, num_comp, sum_comp, codes_info, col_types, \
                    output_cols, index_to_d2v, year, month)

            # add rows for nontop50 trcns
            missing_trcns = list(all_trcn_codes - set(codes_info[0]))
            for tracon in missing_trcns:
                index_id = f"{tracon} {int(year)}/{int(month)}"
                index_to_d2v[index_id] = mis_row

        fin = pd.DataFrame.from_dict(index_to_d2v, orient='index')
        month_range_dict[month_range] = month_range_dict.get(month_range, []) + [fin]

def load_replace_dictionary(col):
    """
    Loads dictionary that maps abbreviations to their full form
    @param: col (str): narrative/synopsis/combined/narrative_synopsis_combined
        which string we are currently analyzing
    @returns: replace_dict (dict): abbreviation -> fullform
    """
    total_cts = pd.read_csv(f'results/total_cts_tagged_{col}.csv', index_col=0)
    total_cts = total_cts.loc[total_cts['abrev'] == 1, :]

    # generate replace dictionary
    replace_dict = {}
    for _, row in total_cts.iterrows():
        for dict_name in ['nasa', 'faa', 'casa', 'hand', 'iata']:
            dict_fullform = row[f'{dict_name}_fullform']
            if not pd.isna(dict_fullform):
                dict_fullform = str(dict_fullform).lower()
                replace_dict[row['acronym']] = dict_fullform
                break
    return replace_dict

def incident_unique_codes():
    """
    Loads unique codes from NTSB/FAA incident/accident dataset
    @returns: np.ndarray of unique codes from the incident/accident dataset
    """
    unique_code_fn = '../results/unique_airport_code_ntsb_faa.pckl'
    unique_ntsb_faa_codes = pickle.load(open(unique_code_fn, 'rb'))
    return unique_ntsb_faa_codes

def all_unique_codes(tracon_month_unique, unique_codes):
    """
    Combines the unique codes from the ASRS dataset with the tracon_codes from the
    incident/accident dataset.
    @param: tracon_month_unique (pd.DataFrame) of unique tracon_code/year/month combinations
        from asrs dataset
    @param: unique_codes (np.ndarray) of unique tracon_codes from incident/accident dataset
    @returns: np.ndarray of unique tracon_codes from either incident/accident dataset or ASRS
        dataset
    """
    set_unique_codes = set(unique_codes)

    asrs_added_tracons = []
    for tracon_code in tracon_month_unique['tracon_code'].unique():
        if tracon_code not in set_unique_codes:
            asrs_added_tracons.append(tracon_code)

    return np.hstack([unique_codes, np.array(asrs_added_tracons)])

def filter_top50(unique_ntsb_faa_codes, tracon_month_unique):
    """
    Only selects top50 iata codes from both parameters
    @param: unique_ntsb_faa_codes (np.ndarray of str) of unique tracon_codes from either dataset
    @param: tracon_month_unique (pd.DataFrame) w/columns tracon_code/year/month of ASRS dataset
    """
    top_50_iata = \
            set(pd.read_excel('../datasets/2010 Busiest Airports wikipedia.xlsx')['IATA'].iloc[1:])
    unique_ntsb_faa_codes = np.apply_along_axis(\
            lambda x: [elem for elem in x if elem in top_50_iata], \
            0, \
            unique_ntsb_faa_codes)
    tracon_month_unique = tracon_month_unique.loc[\
        tracon_month_unique['tracon_code'].apply(lambda x: x in top_50_iata)]
    return unique_ntsb_faa_codes, tracon_month_unique

def add_missing_rows(unique_ntsb_faa_codes, tracon_month_unique):
    """
    This adds the trcns from NTSB/FAA incident/accident dataset to that of the ASRS trcns
    @param: unique_ntsb_faa_codes (np.ndarray of str) of unique tracon_codes from either dataset
    @param: tracon_month_unique (pd.DataFrame) w/columns tracon_code/year/month of ASRS dataset
    @returns: tracon_month_unique with rows from ASRS dataset and FAA/NTSB incident/accident dataset
    """
    all_combs = set(tracon_month_unique.apply(lambda x: (x[0], x[1], x[2]), axis=1))
    added_rows = {'tracon_code': [], 'month': [], 'year':[]}
    for tracon, month, year in product(unique_ntsb_faa_codes, range(1, 13), range(1988, 2020)):
        if (tracon, month, year) not in all_combs:
            added_rows['tracon_code'].append(tracon)
            added_rows['month'].append(month)
            added_rows['year'].append(year)

    return tracon_month_unique.append(pd.DataFrame.from_dict(added_rows))

def process_with_replace(field, r_d):
    """
    This pre-processes a field by replacing words using the replace_dict.
    @param: field (str) string to be processed
    @param: r_d (dict[abbrev] -> fullform) dictionaray that maps abbreviation to fullform
    @returns: processed field
    """
    np_res = preprocess_helper.replace_words(str(field), replace_dict=r_d)
    return ' '.join(np_res)

def generate_tagged_docs(np_fields, r_d):
    """
    This generates a list of tagged documents, which gets fed into Doc2Vec model. Each document
    is given a tag, and we generate a dictionary that maps document to the tag
    @param: np_fields (iterable, usually np.ndarray of str) of all the strings we need to analyze
    @param: r_d (dict[abbrev] -> fullform) dictionaray that maps abbreviation to fullform
    @returns: docs (list of TaggedDocument) representing the documents we wish to analyze
    @returns: doc_to_idx (dict[doc] -> tag) field that maps documents to their tag
    """
    # creating list of tagged documents
    docs = []
    doc_to_idx = {}
    ctr = 0
    for field in tqdm(np_fields, total=np_fields.shape[0]):
        doc_str = process_with_replace(field, r_d)
        if doc_str not in doc_to_idx:
            doc_to_idx[doc_str] = ctr
            docs.append(TaggedDocument(doc_str, [ctr]))
            ctr += 1
    return docs, doc_to_idx

def d2v_multiple_reports(all_pds):
    """
    For columns with multiple reports (narrative/callback), we calculate the cos_sim between
    the reports and save the result as a new column.
    @param: all_pds (pd.DataFrame) ASRS dataset
    @returns: all_pds (pd.DataFrame) w/new columns for cos_sim between reports
    """
    for mult_col in ['narrative', 'callback']:
        reps = np.hstack((all_pds[f'{mult_col}_report1'].unique(),\
                all_pds[f'{mult_col}_report2'].unique()))
        reps = reps.astype(str)

        for r_d in [load_replace_dictionary(mult_col), {}]:
            replace = len(r_d) > 0
            cos_col_name = f'{mult_col}_multiple_reports_cos_sim{"_ff" if replace else ""}'
            all_pds[cos_col_name] = np.nan

            docs, doc_to_idx = generate_tagged_docs(reps, r_d)

            # train doc2vec
            model = Doc2Vec(docs, vector_size=20, window=3)
            only_mult_rep_df = all_pds.loc[all_pds[f'{mult_col}_multiple_reports'], :]
            for idx, row in tqdm(only_mult_rep_df.iterrows(), total=only_mult_rep_df.shape[0]):
                report1 = process_with_replace(row[f'{mult_col}_report1'], r_d)
                report2 = process_with_replace(row[f'{mult_col}_report2'], r_d)

                vec1 = model.docvecs[doc_to_idx[report1]]
                vec2 = model.docvecs[doc_to_idx[report2]]

                cos_sim = cosine_similarity(vec1.reshape(1, 20), vec2.reshape(1, 20))
                all_pds.loc[idx, cos_col_name] = (cos_sim[0, 0] + 1) / 2
    return all_pds

def cos_sim_analysis(all_pds, tracon_month_unique, lag=1):
    """
    This performs the cos_sim calculation for all columns (and adds the columns to the dataset).
    @param: all_pds (pd.DataFrame) ASRS dataset
    @param: tracon_month_unique (pd.DataFrame) with only columns = tracon_code/year/month
        that lists out unique tracon_months within dataset
    """
    for col in ['narrative', 'synopsis', 'callback', 'combined', 'narrative_synopsis_combined']:
        month_range_dict = {}
        for r_d in [load_replace_dictionary(col), {}]:
            reps = all_pds[col].unique()
            docs, doc_to_idx = generate_tagged_docs(reps, r_d)
            # train doc2vec
            print('training doc2vec models. This can take a while...')
            model = Doc2Vec(docs, vector_size=20, window=3)

            analyze_d2v(all_pds, model, len(r_d) > 0, month_range_dict, col=col, \
                    field_dict=doc_to_idx, tracon_month_unique=tracon_month_unique, \
                    replace_dict=r_d, lag=lag)

        for month_range in month_range_dict:
            res = pd.concat(month_range_dict[month_range], axis=1)
            res.to_csv(f'results/d2v_tracon_month_{col}_{month_range}mon_{lag}lag.csv')

def main():
    """
    Calculates average cosine similarity for each tracon_month for all possible permutations
    """
    # load files
    all_pds = preprocess_helper.load_asrs(load_saved=True)
    all_pds = all_pds.reset_index().drop('index', axis=1)
    # all_pds = preprocess_helper.tracon_analysis(all_pds)

    # top 50/missing row analysis
    tracon_month_unique = all_pds[['tracon_code', 'month', 'year']].drop_duplicates()
    unique_codes = incident_unique_codes()
    unique_ntsb_faa_codes = all_unique_codes(tracon_month_unique, unique_codes)
    print('after missing row analysis')

    unique_ntsb_faa_codes, tracon_month_unique = \
            filter_top50(unique_ntsb_faa_codes, tracon_month_unique)
    tracon_month_unique = add_missing_rows(unique_ntsb_faa_codes, tracon_month_unique)
    print('after missing rows')

    # all_pds = d2v_multiple_reports(all_pds)
    cos_sim_analysis(all_pds, tracon_month_unique, lag)

if __name__ == "__main__":
    main()
