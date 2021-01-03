import urllib.request as request
from urllib.parse import quote_plus, quote
from collections import namedtuple

from tqdm import tqdm
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

coverage = namedtuple('coverage', ['part', 'total'])
us_state_abbrev = pd.read_csv('datasets/us_state_abbrev.csv', index_col = 0)
us_state_abbrev = us_state_abbrev.to_dict()['full']

def get_city(x):
    x_split = x.split(",")
    if len(x_split) != 2 and len(x_split) != 3:
        return np.nan
    else:
        return x_split[0].lower().strip()

def get_state(x, full = True):
    x_split = x.split(",")
    if len(x_split) == 3:
        second_elem = x_split[1].strip()
        if full:
            return us_state_abbrev.get(second_elem, second_elem)
        return second_elem
    elif len(x_split) == 2:
        return us_state_abbrev.get(x_split[-1].strip(), np.nan)
    else:
        return np.nan # if 2, only city, country

def get_country(x):
    x_split = x.split(",")
    if len(x_split) >= 2:
        last_elem = x_split[-1].strip()
        if last_elem in us_state_abbrev:
            return "United States"
        else:
            if last_elem == "USA":
                return "United States"
            return last_elem
    else:
        return np.nan
