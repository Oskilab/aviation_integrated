# Requirements
You will need the following python packages to run this project.
* pandas
* numpy
* ftfy
* IPython

# Running
```
./run_all
```

# Files
* `name_code_clean.py` and `name_code_clean.ipynb`: this file cleans the `ntsb_airportcode.csv` and `ntsb_airportname.csv` using information found from `airports.csv`
    * using multiple cleaning passes, the code creates multiple dictionaries saved at `name_dict.npy`, `code_dict.npy` and `code_dict_risky.npy`
    * Cleaning is accomplished by removing whitespace, uppercasing, etc. If airport codes are not found, some pre-processing steps are done to see if a slightly adjusted code is found (removing first letter, etc.)
    * The `name_dict.npy` dictionary maps airport names to the index at which the airport name first appears in `airports.csv`whereas the `code_dict.npy` maps airport codes to a list of indices in which the airport code appears in `airports.csv`
    * `code_dict_risky.npy` utilizes a riskier version of cleaning
    * The dictionaries are used later in `merge_data.py`
* `merge_data.py` and `merge_data.ipynb`: this file fills in the missing airport names and codes found in the `NTSB_AIDS_full.txt` dataset (represents the accidents/incidents in NTSB dataset)
    * It utilizes the latitude/longitude information (`airports.csv`) to fill in invalid rows (of airport names/codes) in the `NTSB_AIDS_full.txt`.
    * It creates new columns (`airportcode_new`, `airportname_new`, `code_valid`, `name_valid`). The `{code|name}_valid` columns can be `trusted`, `risky`, `notfound` (see `merge_data.py` for more details)
