import pandas as pd, numpy as np, re
from IPython import embed
from tqdm import tqdm

casa = pd.read_csv('dictionaries/CASA.csv')
faa = pd.read_csv('dictionaries/FAA.csv')
iata_iaco = pd.read_csv('dictionaries/IATA_IACO.csv', encoding = 'cp437')
asrs = pd.read_csv('datasets/ASRS 1988-2019_extracted.csv')

# casa remove 'xxx (xxx)' parenthesis
casa['acronym'] = casa['acronym'].str.replace('\(.+\)', '')

iata_iaco = iata_iaco[~iata_iaco['acronym'].isna()]

assert(iata_iaco['acronym'][~iata_iaco['acronym'].str.islower()].shape[0] == 0)
assert(faa['acronym'][~faa['acronym'].str.islower()].shape[0] == 0)
assert(casa['acronym'][~casa['acronym'].str.islower()].shape[0] == 0)

def convert_to_words(row, col = 'narrative'):
    s = row[col]
    if isinstance(s, float) and np.isnan(s):
        s = ''
    s = s.lower()
    res = np.array(re.split('[( | ;|\. |\.$]', s))
    res = res[res != ''].flatten()
    return res

tolist = lambda x: [x['synopsis'], x['narrative']]
asrs['combined'] = asrs.apply(tolist, axis = 1).str.join(sep = ' ')


split = asrs.apply(lambda x: convert_to_words(x, "narrative"), axis = 1)
d = {}
for elem in split:
    for word in elem:
        if word in d:
            d[word] += 1
        else:
            d[word] = 1
embed()
