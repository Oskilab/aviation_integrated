import pandas as pd, numpy as np, enchant, nltk, itertools, re
from nltk.corpus import stopwords
from IPython import embed
from collections import Counter
import matplotlib.pyplot as plt
from preprocess_helper import load_asrs, create_counter, load_dictionaries
from preprocess_helper import neg_nonword_to_neg_word_set, neg_nonword_to_airport_set, \
        potential_words_from_negnw
from bs4 import BeautifulSoup
import argparse

summary_cols = ['total words', 'unique words', 'neg_nonword unprocessed', 'unique neg_nonword unprocessed',\
        'neg_hand2', 'unique neg_hand2','airport', 'unique airport', 'neg_nonalpha', 'unique neg_nonalpha', \
        'neg_city', 'unique neg_city','neg_nonword processed', 'unique neg_nonword procesed',  \
        'pot. words frm neg_nw', 'unique pot. words frm neg_nw','top down', 'unique top down','hand', \
        'unique hand' 'hand2', 'unique_hand2']

def parse_args():
    """
    Parses commandline arguments. If -t isincluded, we are only testing the pipeline,
    and a subset of the data is taken for fast computations
    @returns: test (bool) True if -t is included, False otherwise
    """
    parser = argparse.ArgumentParser(description='Analyze abbreviations.')
    parser.add_argument('-t', action = 'store_true')
    args = parser.parse_args()

    test = args.t
    return test

def parse_city_words():
    """
    This reads an html file of a list of US cities, and it creates a set of all
    words that are utilized in a city name (e.g., 'york' from 'New York')
    @returns: cities_words (set of str)
    """
    # get list of cities
    with open('datasets/List_of_United_States_cities_by_population') as wiki_cities:
        list_of_cities = BeautifulSoup(wiki_cities.read(), 'html.parser')

    # process html
    cities = []
    for x in list_of_cities.find_all('td'):
        try:
            if x.a.string is not None:
                cities.append(x.a.string)
        except AttributeError:
            continue

    # only use subset
    ny_ind, wood_ind = cities.index('New York'), cities.index('Woodbridge')
    cities = cities[ny_ind:wood_ind]

    cities_words = []
    for x in cities:
        for elem in x.split(" "):
            cities_words.append(elem.lower())

    return set(cities_words)

def get_all_abrevs(aviation_dicts):
    """
    Generates a set of all known abbreviations (top-down).
    @param: aviation_dicts (dict[abbrev_dataset_name] -> pd.DataFrame), see
        preprocess_helper.py for more information
    @returns: set of all known abbreviations (top-down)
    """
    # create set of all abbreviations in all dictionaries
    dfs = list(aviation_dicts.values())
    all_abrevs = set(dfs[0]['acronym'])
    for i in range(1, len(dfs)):
        all_abrevs.update(dfs[i]['acronym'])
    return all_abrevs

def negatives(total_cts, summary_dict, word_categories, orig_col):
    """
    This analyzes the words of the ASRS dataset that are not found in any
    abbreviation_dictionary. We label each word with a specific tag such as
    neg_word, neg_nonword, ..., to organize all known "negative" words.
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: summary_dict (dict[summary_colname] -> value), this keeps track
        of global information that would be useful (the number of neg_nonwords, 
        or the nubmer of neg_words, etc.)
    @param: word_categories (list of objects) includes information on important
        types of word_categories
        word_categories[0] = set of all known top-down abbreviations
        word_categories[1] = set of all english stop words
        word_categories[2] = enchant dictionary (see pyenchant module) 
            functions as an english dictionary
        word_categories[3] = set of all neg_nonwords that are actually english words
            this is handcoded and added back to neg_word_hand2
            see preprocess_helper:neg_nonword_to_neg_word_set()
        word_categories[4] = set of all neg_nonwords that are actually part of
            airport names. These are added to neg_airport category
        word_categories[5] = set of all words from city names (scraped from wiki)
            see parse_city_words() for more info
    @param: orig_col (str) column that we are analyzing
    @returns: portion of total_cts that are negative words (words that don't occur
        in any aviation dictionary). Each word is given an associated tag
        as well as abrev = 0 (when it's not labelled as an abbreviation), or 1 when it is
    """
    global summary_cols
    all_abrevs, eng_stopwords, d, negnw_to_negw, negnw_to_airport, cities = word_categories

    # Negatives (or words that are not found in any abbreviation dictionary)
    fn = total_cts.loc[total_cts.index.map(lambda x: x not in all_abrevs), :].copy()
    fn.index = fn.index.map(str)
    fn['abrev'] = 0
    fn['tag'] = ''

    # categorize negatives by whether or not the words are included/excluded in certain
    # dictionaries. For full description of all tags, see top-level README.md
    fn.loc[fn.index.map(lambda x: x in eng_stopwords), 'tag'] = 'neg_stopword'
    fn.loc[fn.index.map(lambda x: x not in eng_stopwords and d.check(x)), 'tag'] = \
            'neg_word'

    sel = fn.index.map(lambda x: x not in eng_stopwords and not d.check(x) and x not in cities)
    fn.loc[sel, 'abrev'] = 1
    fn.loc[sel, 'tag'] = 'neg_nonword'
    summary_dict[summary_cols[2]] = fn.loc[sel, 0].sum() # neg nonword unprocessed
    summary_dict[summary_cols[3]] = fn[sel].shape[0] # unique neg nonword unprocessed

    # add back handcoded neg_nonwords to neg_words
    neg_word_sel = np.logical_and(sel, fn.index.map(lambda x: x in negnw_to_negw))
    fn.loc[neg_word_sel, 'abrev'] = 0
    fn.loc[neg_word_sel, 'tag'] = 'neg_word_hand2'
    summary_dict[summary_cols[4]] = fn.loc[neg_word_sel, 0].sum() # neg_hand2
    summary_dict[summary_cols[5]] = fn[neg_word_sel].shape[0] # unique neg_hand2

    # convert some neg_nonword to airport tag
    neg_airport_sel = np.logical_and(sel, fn.index.map(lambda x: x in negnw_to_airport))
    fn.loc[neg_airport_sel, 'abrev'] = 0
    fn.loc[neg_airport_sel, 'tag'] = 'neg_airport'
    summary_dict[summary_cols[6]] = fn.loc[neg_airport_sel, 0].sum() # neg_hand2
    summary_dict[summary_cols[7]] = fn[neg_airport_sel].shape[0] # unique neg_hand2

    sel = np.array(sel) & fn.index.str.contains('[^A-Za-z]', na = True)
    fn.loc[sel, 'abrev'] = 0
    fn.loc[sel, 'tag'] = 'neg_nonalpha_abrev'
    summary_dict[summary_cols[8]] = fn.loc[sel, 0].sum() # neg_nonalpha
    summary_dict[summary_cols[9]] = fn[sel].shape[0] # unique neg_nonalpha

    sel = fn.index.map(lambda x: x not in eng_stopwords and not d.check(x) and x in cities)
    fn.loc[sel, 'abrev'] = 0
    fn.loc[sel, 'tag'] = 'neg_nonword_city_exception'
    summary_dict[summary_cols[10]] = fn.loc[sel, 0].sum() # city exception
    summary_dict[summary_cols[11]] = fn[sel].shape[0] # unique city exception

    sel = fn['tag'] == 'neg_nonword'
    summary_dict[summary_cols[12]] = fn.loc[sel, 0].sum() # neg_nonword processed
    summary_dict[summary_cols[13]] = fn[sel].shape[0] # unique neg_nonword processed

    pot_words = potential_words_from_negnw()
    sel = fn.index.map(lambda x: x in pot_words)
    summary_dict[summary_cols[14]] = fn.loc[sel, 0].sum() # potential words from neg_nw
    summary_dict[summary_cols[15]] = fn[sel].shape[0] # unique potential words from neg_nw

    fn.to_csv(f'results/fn_tagged_{orig_col}.csv')
    return fn

def positives(total_cts, summary_dict, word_categories, aviation_dicts, orig_col):
    """
    This analyzes the words of the ASRS dataset that are found in any
    abbreviation_dictionary. We label each word with a specific tag such as
    pos_word, pos_nonword, ..., to organize all known "positive" words.
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: summary_dict (dict[summary_colname] -> value), this keeps track
        of global information that would be useful (the number of neg_nonwords, 
        or the nubmer of neg_words, etc.)
    @param: word_categories (list of objects) includes information on important
        types of word_categories
        word_categories[0] = set of all known top-down abbreviations
        word_categories[1] = set of all english stop words
        word_categories[2] = enchant dictionary (see pyenchant module) 
            functions as an english dictionary
        word_categories[3] = set of all neg_nonwords that are actually english words
            this is handcoded and added back to neg_word_hand2
            see preprocess_helper:neg_nonword_to_neg_word_set()
        word_categories[4] = set of all neg_nonwords that are actually part of
            airport names. These are added to neg_airport category
        word_categories[5] = set of all words from city names (scraped from wiki)
            see parse_city_words() for more info
    @param: orig_col (str) column that we are analyzing
    @returns: portion of total_cts that are positive words (words that do occur
        in any aviation dictionary). Each word is given an associated tag
        as well as abrev = 0 (when it's not labelled as an abbreviation), or 1 when it is
    """
    all_abrevs, eng_stopwords, d, negnw_to_negw, negnw_to_airport, cities = word_categories

    # pos_sel = the rows that have at least one non-NAN value
    pos_sel = total_cts['casa_fullform'].isna()
    for col in ['faa_fullform', 'iata_fullform', 'nasa_fullform', 'hand_fullform']:
        pos_sel = pos_sel & (total_cts[col].isna())
    pos_sel = ~pos_sel

    sum_cts = np.sum(total_cts.loc[:, 0])

    # Stopwords dataframe
    sel = pos_sel & total_cts.index.map(lambda x: x in eng_stopwords)
    total_cts.loc[sel, 'tag'] = 'pos_stopword'

    # Not english stopwords and not english words
    sel = pos_sel & total_cts.index.map(lambda x: not d.check(x) and x not in eng_stopwords)
    total_cts.loc[sel, 'tag'] = 'pos_nonword'
    total_cts.loc[sel, 'abrev'] = 1

    # Not english stopwords and english words
    pos_word_sel = pos_sel & total_cts.index.map(lambda x: d.check(x) and x not in eng_stopwords)
    total_cts.loc[pos_word_sel, 'tag'] = 'pos_word'

    # HANDCODED: we look at English words that are actually abbreviations
    # and add it back to our list of abbreviations. Only looked at top 99%
    # of abbreviations
    hc = pd.read_csv('results/abrev_handcoded.csv', index_col = 0)
    hc_abbrev = set(hc.index)
    sel = pos_sel & total_cts.index.map(lambda x: x in hc_abbrev)
    total_cts.loc[sel, 'tag'] = 'pos_handcoded_abrev'
    total_cts.loc[sel, 'abrev'] = 1

    # remove words that are only found in iata
    sel = pos_word_sel & (~total_cts['iata_fullform'].isna()) & \
            (total_cts['casa_fullform'].isna()) & \
            (total_cts['faa_fullform'].isna())

    # iata_only = words_and_abrev1.loc[sel].copy()
    # words_and_abrev2 = words_and_abrev1.loc[~sel].copy()
    total_cts.loc[sel, 'tag'] = 'pos_iata_only_words'
    total_cts.loc[sel, 'abrev'] = 0
    total_cts = total_cts.reset_index().rename({'index': 'acronym'}, axis = 1)

    top_down_sel = (total_cts['tag'] == 'pos_handcoded_abrev') | (total_cts['tag'] == 'pos_nonword')
    summary_dict[summary_cols[16]] = total_cts.loc[top_down_sel, 0].sum() # top down
    summary_dict[summary_cols[17]] = total_cts[top_down_sel].shape[0] # unique top down
    total_cts.to_csv(f'results/total_cts_tagged_{orig_col}.csv')

def get_top_down_subset(total_cts, dictionary_names):
    """
    Returns all rows in total_cts that are associated with a word that is found
    in any aviation dictionary
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'hand2', 'iata']
    @returns: subset of total_cts with word that is found in any aviation dict
    """
    sel = None
    for dictionary in dictionary_names:
        if sel is None:
            sel = ~total_cts[f'{dictionary}_fullform'].isna()
        else:
            sel = sel | (~total_cts[f'{dictionary}_fullform'].isna())
    subset = total_cts.loc[sel, :].copy()
    return subset

def generate_dictionary_summary(total_cts, dictionary_names, summary_dict):
    """
    This generates a python dict that names dictionary_names (casa/nasa/...)
    to a pd.Series that summarizes the number of times that a word in that
    dictionary showed up in the ASRS dataset.
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'hand2', 'iata']
    @param: summary_dict (dict[summary_col] -> float) contains summary information
        about top_down/bottom_up analysis of words.
    @returns: dictionary_summary (dict[dictionary_name] -> pd.Series of summary info)
    """
    dictionary_summary = {}
    for dictionary in dictionary_names:
        subset = total_cts.loc[~total_cts[f'{dictionary}_fullform'].isna(), :]
        dictionary_summary[dictionary] = pd.Series({'ct': np.sum(subset[0]), \
                'unique_ct': subset.shape[0]})

        if dictionary == 'hand' or dictionary == 'hand2':
            summary_dict[f'{dictionary}'] = subset[0].sum()
            summary_dict[f'unique {dictionary}'] = subset[0].shape[0]
    return dictionary_summary

def generate_bar_chart(total_cts, agg_func, output_fn, title="", abrev=True):
    """
    This generates a barchart of total_cts split by tag (which we generated).
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: agg_func (func) used to aggregate counts
    @param: output_fn (str) filepath for output
    @param: title (str) how to title the bar chart
    @param: abrev (bool) if True, we are only analyzing the portion of total_cts
        that are labelled as abbreviations. Otherwise, the full ds
    """
    if abrev:
        tmp = total_cts.loc[total_cts['abrev'] == 1, [0, 'tag']].groupby('tag').agg(agg_func)
    else:
        tmp = total_cts.loc[:, [0, 'tag']].groupby('tag').agg(agg_func)
    tot = np.sum(tmp)[0]

    ax = (tmp / tot).plot.barh(y = 0, figsize = (10, 6), title=title)
    plt.tight_layout()
    ax.get_figure().savefig(output_fn)

def generate_all_bar_charts(total_cts, orig_col):
    """
    The following section creates a series of bar plots showing the breakdown of the total corpus
    by the created tags. If there's a unique preceding the name of the file, then it's the breakdown
    of unique words (or the vocabulary used in the dataset)
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: orig_col (str) column we are analyzing
    """
    generate_bar_chart(total_cts, np.sum, f'results/abrev_bar_plot_{orig_col}.png', title=\
            f"Abrev Bar Chart (Prop of Abrevs, {orig_col})")

    generate_bar_chart(total_cts, lambda x: np.unique(x).shape[0], \
            f'results/unique_abrev_bar_plot_{orig_col}.png', 
            title=f"Unique Abrev Bar Chart (Prop. of Abrev vocab, {orig_col})")

    generate_bar_chart(total_cts, np.sum, f'results/corpus_bar_plot_{orig_col}.png', 
            title=f"Corpus Bar Chart (Prop. of Total Corpus, {orig_col})", abrev=False)

    generate_bar_chart(total_cts, lambda x: np.unique(x).shape[0], \
            f'results/unique_corpus_bar_plot_{orig_col}.png', 
            title=f"Unique Corpus Bar Chart (Prop. of total vocab, {orig_col})", abrev=False)

def generate_abrev_tag_summary(total_cts, orig_col):
    """
    This summarizes total_cts and breaks it down by tag to see what proportion of the
    dataset is from each tag. Only looks at abbreviations.
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: orig_col (str) column we  are analyzing
    """
    only_abrevs = total_cts.loc[total_cts['abrev'] == 1]

    # create summary of all tags by grouping by tags, and dividing by the sum of the counts
    ct_summary = only_abrevs[['tag', 0]].groupby(['tag']).sum()
    total_num = np.sum(ct_summary[0])
    ct_summary.loc[:, 'prop_of_ct'] = ct_summary.loc[:, 0] / total_num
    ct_summary.rename({0: 'ct'}, axis = 1,inplace = True)

    unique_summary = only_abrevs[['tag', 0]].groupby(['tag']).count()
    total_num = np.sum(unique_summary[0])
    unique_summary.loc[:, 'prop_of_unique'] = unique_summary.loc[:, 0] / total_num
    unique_summary.rename({0: 'unique_ct'}, axis = 1,inplace = True)

    total_summary = unique_summary.merge(ct_summary, on = 'tag')
    total_summary.to_csv(f'results/tag_summary_{orig_col}.csv')


def process_asrs(total_cts, fn, aviation_dicts):
    """
    This combines top_down analysis (utilizing aviation dicts) and bottom_up analysis
    (looking at all words not in aviation dicts). Also merges with aviation dictionaries
    to get fullform of abbreviations
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: fn (pd.DataFrame) result of negatives function
    @param: aviation_dicts (dict[abbrev_dataset_name] -> pd.DataFrame), see
        preprocess_helper.py for more information
    @returns: processed total_cts (pd.DataFrame) with adjusted columns for fullforms
    """
    total_cts = total_cts.reset_index().rename({'index': 'acronym'}, axis = 1)
    for aviation_pd in aviation_dicts.values():
        total_cts = total_cts.merge(aviation_pd, on = "acronym", how = "left")

    total_cts.drop_duplicates(inplace = True)
    total_cts.set_index('acronym', inplace = True)

    total_cts['abrev'] = 0
    total_cts['tag'] = ''
    total_cts = pd.concat([fn, total_cts], axis = 0, ignore_index = False, sort = False)
    total_cts.sort_values(by = 0, ascending = False, inplace = True)
    return total_cts

def analyze_dictionary_words(total_cts, summary_dict, orig_col):
    """
    This generates a summary of the column we are analyzing by breaking down the 
    abbreviations by aviation_dictionary. This is saved to results/ subfolder
    @param: total_cts (pd.DataFrame) of words -> # times the word occurs
        in ASRS dataset. The column named 0 is the count
    @param: summary_dict (dict[summary_col] -> float) contains summary information
        about top_down/bottom_up analysis of words.
    @param: orig_col (str) column we are analyzing
    """
    dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'hand2', 'iata']
    dictionary_summary = generate_dictionary_summary(total_cts, dictionary_names, summary_dict)

    dictionary_names.remove('iata')
    subset = get_top_down_subset(total_cts, dictionary_names)

    dictionary_summary['total_top_down_abrev'] = pd.Series({'ct': np.sum(subset[0]), \
            'unique_ct': subset.shape[0]})
    dictionary_summary = pd.DataFrame.from_dict(dictionary_summary, orient = 'index')
    dictionary_summary.to_csv(f'results/dictionary_summary_{orig_col}.csv')

def analyze_column(asrs, orig_col, summary_pd, word_categories, aviation_dicts):
    """
    This analyzes the ASRS dataset by utilizing aviation dictionaries and word categories
    to split all known words into different categories.
    @param: ASRS (pd.DataFrame) dataset generated from preprocess_asrs
    @param: orig_col (str) column we are analyzing
    @param: summary_pd (dict[summary_col] -> summary_metric), for summarizing the 
        full pipeline
    @param: word_categories (list of objects) includes information on important
        types of word_categories
        word_categories[0] = set of all known top-down abbreviations
        word_categories[1] = set of all english stop words
        word_categories[2] = enchant dictionary (see pyenchant module) 
            functions as an english dictionary
        word_categories[3] = set of all neg_nonwords that are actually english words
            this is handcoded and added back to neg_word_hand2
            see preprocess_helper:neg_nonword_to_neg_word_set()
        word_categories[4] = set of all neg_nonwords that are actually part of
            airport names. These are added to neg_airport category
        word_categories[5] = set of all words from city names (scraped from wiki)
            see parse_city_words() for more info
    @param: aviation_dicts (dict[abbrev_dataset_name] -> pd.DataFrame), see
        preprocess_helper.py for more information
    """
    summary_dict = {}
    # this creates a dataframe of word counts
    total_cts = create_counter(asrs, orig_col)
    total_cts.sort_values(by = 0, ascending = False, inplace = True)

    summary_dict[summary_cols[0]] = total_cts[0].sum() # total words
    summary_dict[summary_cols[1]] = total_cts[0].shape[0] # unique words

    # analyze words not found in aviation dictionaries
    fn = negatives(total_cts, summary_dict, word_categories, orig_col)

    total_cts = process_asrs(total_cts, fn, aviation_dicts)

    # Positives (or words that are found in an abbreviation dictionary)
    # combine with full-form from dictionaries
    positives(total_cts, summary_dict, word_categories, aviation_dicts, orig_col)

    analyze_dictionary_words(total_cts, summary_dict, orig_col)

    summary_pd[orig_col] = pd.Series(summary_dict)

    # generate bar charts of distribution
    generate_all_bar_charts(total_cts, orig_col)

    # save tags
    for tag in total_cts['tag'].unique():
        total_cts.loc[total_cts['tag'] == tag, :].to_csv(f"results/{orig_col}_{tag}.csv")

    # save summary dataframe of tags (only abbreviations)
    generate_abrev_tag_summary(total_cts, orig_col)

def main():
    test = parse_args()
    cities = parse_city_words()

    # load datasets
    aviation_dicts = load_dictionaries() # see preprocess helper
    all_abrevs = get_all_abrevs(aviation_dicts)
    asrs = load_asrs(load_saved = False, test = test) # see preprocess helper

    # python dictionaries
    eng_stopwords = set(stopwords.words('english'))
    d = enchant.Dict("en_US")

    # load neg nonword sets
    negnw_to_negw = neg_nonword_to_neg_word_set()
    negnw_to_airport = neg_nonword_to_airport_set()

    word_categories = [all_abrevs, eng_stopwords, d, negnw_to_negw, negnw_to_airport, cities]

    # analyze each column
    summary_pd = {}
    for orig_col in ['narrative', 'synopsis', 'callback' , 'combined', 'narrative_synopsis_combined']:
        analyze_column(asrs, orig_col, summary_pd, word_categories, aviation_dicts)

    # save summary of all columns and corresponding dictionaries/abbreviations/etc.
    all_summary = pd.DataFrame.from_dict(summary_pd, orient = 'index')
    all_summary.to_csv('results/all_summary.csv')

if __name__ == "__main__":
    main()
