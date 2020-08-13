from IPython import embed
import pandas as pd, numpy as np
import re

nonalpha = pd.read_csv('results/neg_nonalpha_abrev.csv', index_col = 0)
tot = np.sum(nonalpha.loc[:, "0"])

# starts with (, ' or [
r1 = "^([\(\'\[]{1})([A-Za-z\d]{1,})$"
sel1 = nonalpha['acronym'].str.match(r1)
res1 = nonalpha.loc[sel1, :]
ct1 = np.sum(res1.loc[:, "0"]) / tot
print(f"regex pattern 1 {ct1 * 100:.2f}%")

# ends with ), ' or ]
r2 = "^([A-Za-z\d]{1,})([\)\'\]]{1})$"
sel2 = nonalpha['acronym'].str.match(r2)
res2 = nonalpha.loc[sel2, :]
ct2 = np.sum(res2.loc[:, "0"]) / tot
print(f"regex pattern 2 {ct2 * 100:.2f}%")

# starts with (, ' or [ and ends with ), ', ]
r3 = "^([\(\'\[]{1})([A-Za-z\d]{1,})([\(\'\[]{1})$"
sel3 = nonalpha['acronym'].str.match(r3)
res3 = nonalpha.loc[sel3, :]
ct3 = np.sum(res3.loc[:, "0"]) / tot
print(f"regex pattern 3 {ct3 * 100:.2f}%")

# ends with ? or :, and only has alphabetical characters
r4 = "^([A-Za-z]{1,})([\?:])$"
sel4 = nonalpha['acronym'].str.match(r4)
res4 = nonalpha.loc[sel4, :]
ct4 = np.sum(res4.loc[:, "0"]) / tot
print(f"regex pattern 4 {ct4 * 100:.2f}%")

# only alphabetical with a / in the middle
r5 = "^([A-Za-z]{1,})/([A-Za-z]{1,})$"
sel5 = nonalpha['acronym'].str.match(r5)
res5 = nonalpha.loc[sel5, :]
ct5 = np.sum(res5.loc[:, "0"]) / tot
print(f"regex pattern 5 {ct5 * 100:.2f}%")

all_sel = [sel1, sel2, sel3, sel4, sel5]
all_sel = sel1 | sel2 | sel3 | sel4 | sel5
embed()
