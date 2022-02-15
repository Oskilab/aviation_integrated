# Correction of the code
1. Line 14 of `preprocess_helper.py`: "/" to "\\".
2. Line 44 of `abbrev_words_analysis.py`: add `encoding="utf8"` in the `open()` sentence.
3. Line 193 of `find_ntsb_code.py`: `code` to `Code` on the LHS of the assignment. 

# Addition in the code
1. In `tracon_analysis()` of `preprocess_helper.py`, in line from 166 to 172 and from 183 to 185,
add the functionality of counting the number of `atcadisory` that are split - `AtcAdvisoryMultCount`.

# Update
* Update in 01/04/2022: Discard CASA dictionary temporarily and stop testing spelling check.
* Update in 01/07/2022: Use the top 30 airports listed in `FAA Core Airports 2014.csv`.
* Update in 01/07/2022: Use the top 30 airports listed in `FAA Core Airports 2014.csv`.