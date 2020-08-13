#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import ftfy
from IPython import embed

"""
Phase 1: Clean airportcode and airportname from ntsb. We strip whitespace, uppercase, etc.
"""
ntsb_code = pd.read_csv("datasets/NTSB_airportcode.csv")
ntsb_name = pd.read_csv("datasets/NTSB_airportname.csv")
airport_code = pd.read_csv("datasets/airports.csv")

def clean(col): 
    toReturn = []
    for string in col:
        toReturn.append(''.join([c.upper() for c in string if c.isalpha() or c.isnumeric()]))
    return toReturn

def fix_unicode(iterable):
    iterable_org = [str(d) for d in iterable]
    iterable_fix_unicode, iterable_failed = [], []
    for elem in iterable_org:
        try:
            iterable_fix_unicode.append(ftfy.fix_text(elem))
        except:
            iterable_failed.append(str(elem))
    assert(len(iterable_failed) == 0)
    return iterable_fix_unicode

# Fix unicode 
code_fix_unicode = fix_unicode(ntsb_code["airportcode"])
name_fix_unicode = fix_unicode(ntsb_name["airportname"])

ntsb_code["airportcode"] = clean(code_fix_unicode)
ntsb_name["airportname"] = [string.strip() for string in name_fix_unicode]
# airport_code['ident'] = [string.upper().strip() for string in airport_code['ident']]
airport_code['iata_code'] = [str(string).upper().strip() for string in airport_code['iata_code']]


"""
Phase 2: Create a dictionary for airportcode -> index in dataframe
Example: '5Y7' -> 6747 implies that airportcode 5Y7 occurs at index 6747 of airports.csv
"""

def split_using_set(iterable, match_set):
    matched, not_matched = [], []
    for elem in iterable:
        if elem in match_set:
            matched.append(elem)
        else:
            not_matched.append(elem)
    return matched, not_matched
# ### Build a dic for code
# round 1
code_reference = [string.upper().strip() for string in set(airport_code['iata_code'])]
code_matched, code_not_matched = split_using_set(set(ntsb_code["airportcode"]), \
        code_reference)
print('matched 1', len(code_matched), len(code_not_matched))

# round 2 - some code in reference have a one-letter prefix

def clean_first_letter(col): 
    toReturn = []
    for string in col: 
        try: 
            toReturn.append(string[1:])
        except: 
            pass
    return toReturn

code_reference2 = set(clean_first_letter(code_reference))
code_matched2, code_not_matched2 = split_using_set(code_not_matched, code_reference2)
print('matched 2', len(code_matched2), len(code_not_matched2))

# round 3 - risky, see if some code in data has its last letter missing

def clean_last_letter(col): 
    toReturn = []
    for string in col: 
        try: 
            toReturn.append(string[:-1])
        except: 
            pass
    return toReturn

code_reference3 = clean_last_letter(code_reference)
code_matched3, code_not_matched3 = split_using_set(code_not_matched2, code_reference3)
print('matched 3', len(code_matched3), len(code_not_matched3))


# Build dictionary (code: index in airport_code)
# trusted: code_matched, code_matched2
# risky: code_matched3

code_dict_trusted = dict()
for code in code_matched: 
    ind = airport_code.index[airport_code['iata_code'] == code].tolist()[0]
    code_dict_trusted.update({code: ind})

clean_first = [string[1:] for string in airport_code['iata_code']]

for code in code_matched2: 
    clean_first_match = [item == code for item in clean_first]
    ind_lst = airport_code.index[clean_first_match].tolist()
    try: 
        ind = ind_lst[0]
        code_dict_trusted.update({code: ind})
    except:
        print(ind_lst)
print("percent of code matched with airport data, trusted: " + str(len(code_dict_trusted) / len(ntsb_code)))


code_dict_risky = dict()
clean_last = [string[:-1] for string in airport_code['iata_code']]
for code in code_matched3: 
    clean_last_match = [item == code for item in clean_last]
    ind_lst = airport_code.index[clean_last_match].tolist()
    try: 
        ind = ind_lst[0]
        code_dict_risky.update({code: ind})
    except:
        print(ind_lst)
print("percent of code matched with airport data, risky: " + str(len(code_dict_risky) / len(ntsb_code)))


"""
Phase 3: Clean the airportname field from ntsb dataset
"""

def clean_paren(col):
    toReturn = []
    for string in col: 
        string = string.upper().strip()
        left = string.find('(')
        right = string.find(')')
        if left != -1 and right != -1: 
            try: 
                new_string = string[:left] + string[right+1:]
            except:
                new_string = string[:left]
            toReturn.append(new_string)
        elif left != -1: 
            new_string = string[:left]
            toReturn.append(new_string)
        else: 
            toReturn.append(string)
    return toReturn


ntsb_name["airportname_cleanedparen"] = clean_paren(name_fix_unicode)
ntsb_name["airportname_cleanedletter"] = clean(clean_paren(name_fix_unicode))

"""
Phase 4: Create a dictionary for airportname -> list of indices in dataframe
Example: 'HAWTHORNEMUNICIPALAIRPORT' -> [24729, 32991] implies that 
    airportname 5Y7 occurs at index 24729 and 32991 of airports.csv
"""
# round 1
name_reference = set(clean(clean_paren(airport_code["name"])))

name_matched = dict()
name_not_matched = []
for airportname in set(ntsb_name["airportname_cleanedletter"]): 
    if airportname in name_reference: 
        name_matched.update({airportname: airportname})
    else: 
        not_match = True
        for ref in name_reference:
            if (airportname in ref) or (ref in airportname): 
                not_match = False
                name_matched.update({airportname: ref})
                break
        if not_match:
            name_not_matched.append(airportname) # THIS IS A MISTAKE, fixed
print('name matched', len(name_matched), len(name_not_matched))

# round 2
# Build dictionary (name_cleanedletter: index in airport_code)
# trusted: name_matched
# risky: code_matched3

clean_letters = clean(clean_paren(airport_code["name"]))

name_dict_trusted = dict()
for name in name_matched: 
    val = name_matched.get(name)
    clean_match = [item == val for item in clean_letters]
    ind_lst = airport_code.index[clean_match].tolist()
    name_dict_trusted.update({name: ind_lst})

np.save('results/name_dict.npy', name_dict_trusted) 
np.save('results/code_dict.npy', code_dict_trusted)
np.save('results/code_dict_risky.npy', code_dict_risky)
