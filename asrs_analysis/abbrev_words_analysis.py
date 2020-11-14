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

parser = argparse.ArgumentParser(description='Analyze abbreviations.')
parser.add_argument('-t', action = 'store_true')
args = parser.parse_args()

test = args.t
print('abbrev_words_analysis test', test)

# get list of cities
with open('datasets/List_of_United_States_cities_by_population') as wiki_cities:
    list_of_cities = BeautifulSoup(wiki_cities.read(),'html.parser')

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
cities = set(cities_words)

aviation_dicts = load_dictionaries() # see preprocess helper
asrs = load_asrs(load_saved = False, test = test) # see preprocess helper
negnw_to_negw = neg_nonword_to_neg_word_set()
negnw_to_airport = neg_nonword_to_airport_set()

# create set of all abbreviations in all dictionaries
dfs = list(aviation_dicts.values())
all_abrevs = set(dfs[0]['acronym'])
for i in range(1, len(dfs)):
    all_abrevs.update(dfs[i]['acronym'])

# python dictionaries
eng_stopwords = set(stopwords.words('english'))
d = enchant.Dict("en_US")

summary_cols = ['total words', 'unique words', 'neg_nonword unprocessed', 'unique neg_nonword unprocessed',\
        'neg_hand2', 'unique neg_hand2','airport', 'unique airport', 'neg_nonalpha', 'unique neg_nonalpha', \
        'neg_city', 'unique neg_city','neg_nonword processed', 'unique neg_nonword procesed',  \
        'pot. words frm neg_nw', 'unique pot. words frm neg_nw','top down', 'unique top down','hand', \
        'unique hand' 'hand2', 'unique_hand2']
summary_pd = {}
for orig_col in ['narrative', 'synopsis', 'callback' , 'combined', 'narrative_synopsis_combined']:
    summary_dict = {}
    # this creates a dataframe of word counts
    total_cts = create_counter(asrs, orig_col)
    total_cts.sort_values(by = 0, ascending = False, inplace = True)

    summary_dict[summary_cols[0]] = total_cts[0].sum() # total words
    summary_dict[summary_cols[1]] = total_cts[0].shape[0] # unique words

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

    # Positives (or words that are found in an abbreviation dictionary)
    # combine with full-form from dictionaries
    total_cts = total_cts.reset_index().rename({'index': 'acronym'}, axis = 1)
    for aviation_pd in aviation_dicts.values():
        total_cts = total_cts.merge(aviation_pd, on = "acronym", how = "left")

    total_cts.drop_duplicates(inplace = True)
    total_cts.set_index('acronym', inplace = True)

    total_cts['abrev'] = 0
    total_cts['tag'] = ''
    total_cts = pd.concat([fn, total_cts], axis = 0, ignore_index = False, sort = False)
    total_cts.sort_values(by = 0, ascending = False, inplace = True)

    # pos_sel = the rows that have at least one non-NAN value
    pos_sel = total_cts['casa_fullform'].isna()
    for col in ['faa_fullform', 'iata_fullform', 'nasa_fullform', 'hand_fullform']:
        pos_sel = pos_sel & (total_cts[col].isna())
    pos_sel = ~pos_sel

    sum_cts = np.sum(total_cts.loc[:, 0])

    # FAA/CASA common abreviations
    # faa_casa_common = total_cts.loc[list(casa_faa_set), :].copy()
    # faa_casa_common['perc_all'] = faa_casa_common[0] / sum_cts
    # faa_casa_common['perc'] = faa_casa_common[0] / np.sum(faa_casa_common[0])
    # faa_casa_common.sort_values(by = 0, inplace = True, ascending =  False)
    # faa_casa_common.to_csv('results/faa_casa_common.csv')

    # Basic Analysis
    tmp = np.cumsum(total_cts.loc[:, 0]) / sum_cts
    for idx in range(tmp.shape[0]):
        if tmp[idx] > 0.9:
            print(f"{idx + 1} words account for 90% of all abbreviations")
            break

    num_words, num_stop = 0, 0
    ct_stop = 0
    for abrev, num in total_cts.iloc[:idx].loc[:, 0].iteritems():
        if d.check(abrev):
            num_words += 1
        if abrev in eng_stopwords:
            num_stop += 1
            ct_stop += num
    # print(f"Of the {idx + 1} most common abbreviations, {num_words} are English words, " \
    #         + f"{num_stop} are English stopwords")
    # print(f"Stopwords account for {ct_stop * 100 / sum_cts:.2f} of all abbreviations")
    # print(f"1000 abbreviations account for {tmp[999]:.4f} of all abbreviations\n")
    #
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


    dictionary_names = ['nasa', 'faa', 'casa', 'hand', 'hand2', 'iata']
    dictionary_summary = {}
    for dictionary in dictionary_names:
        subset = total_cts.loc[~total_cts[f'{dictionary}_fullform'].isna(), :]
        dictionary_summary[dictionary] = pd.Series({'ct': np.sum(subset[0]), \
                'unique_ct': subset.shape[0]})

        if dictionary == 'hand' or dictionary == 'hand2':
            summary_dict[f'{dictionary}'] = subset[0].sum()
            summary_dict[f'unique {dictionary}'] = subset[0].shape[0]
    summary_pd[orig_col] = pd.Series(summary_dict)

    dictionary_names.remove('iata')
    sel = None
    for dictionary in dictionary_names:
        if sel is None:
            sel = ~total_cts[f'{dictionary}_fullform'].isna()
        else:
            sel = sel | (~total_cts[f'{dictionary}_fullform'].isna())
    subset = total_cts.loc[sel, :].copy()
    dictionary_summary['total_top_down_abrev'] = pd.Series({'ct': np.sum(subset[0]), \
            'unique_ct': subset.shape[0]})
    dictionary_summary = pd.DataFrame.from_dict(dictionary_summary, orient = 'index')
    dictionary_summary.to_csv(f'results/dictionary_summary_{orig_col}.csv')

    # The following section creates a series of bar plots showing the breakdown of the total corpus
    # by the created tags. If there's a unique preceding the name of the file, then it's the breakdown
    # of unique words (or the vocabulary used in the dataset)
    # only abbreviations, total_cts
    tmp = total_cts.loc[total_cts['abrev'] == 1, [0, 'tag']].groupby('tag').agg(np.sum)
    tot = np.sum(tmp)[0]

    ax = (tmp / tot).plot.barh(y = 0, figsize = (10, 6), title = f"Abrev Bar Chart (Prop of Abrevs, {orig_col})")
    plt.tight_layout()
    ax.get_figure().savefig(f'results/abrev_bar_plot_{orig_col}.png')

    # only abbreviations, unique
    tmp = total_cts.loc[total_cts['abrev'] == 1, [0, 'tag']].groupby('tag').agg(lambda x: np.unique(x).shape[0])
    tot = np.sum(tmp)[0]

    ax = (tmp / tot).plot.barh(y = 0, figsize = (10, 6), title = f"Unique Abrev Bar Chart (Prop. of Abrev vocab, {orig_col})")
    plt.tight_layout()
    ax.get_figure().savefig(f'results/unique_abrev_bar_plot_{orig_col}.png')

    # whole corpus, total_cts
    tmp = total_cts.loc[:, [0, 'tag']].groupby('tag').agg(np.sum)
    tot = np.sum(tmp)[0]

    ax = (tmp / tot).plot.barh(y = 0, figsize = (10, 6), title = f"Corpus Bar Chart (Prop. of Total Corpus, {orig_col})")
    plt.tight_layout()
    ax.get_figure().savefig(f'results/corpus_bar_plot_{orig_col}.png')

    # whole corpus, unique
    tmp = total_cts.loc[:, [0, 'tag']].groupby('tag').agg(lambda x: np.unique(x).shape[0])
    tot = np.sum(tmp)[0]

    ax = (tmp / tot).plot.barh(y = 0, figsize = (10, 6), title = f"Unique Corpus Bar Chart (Prop. of total vocab, {orig_col})")
    plt.tight_layout()
    ax.get_figure().savefig(f'results/unique_corpus_bar_plot_{orig_col}.png')

    # save tags
    for tag in total_cts['tag'].unique():
        total_cts.loc[total_cts['tag'] == tag, :].to_csv(f"results/{orig_col}_{tag}.csv")

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

all_summary = pd.DataFrame.from_dict(summary_pd, orient = 'index')
all_summary.to_csv('results/all_summary.csv')
embed()
