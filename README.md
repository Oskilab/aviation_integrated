This repository processes the ASRS dataset by extracting the tracon information. Then, the ASRS columns `narrative, synopsis, and narrative` (which are descriptions and summaries of accidents/incidents that we have access to) are all analyzed utilizing LIWC, Doc2Vec, and word categorization. Then, the FAA/NTSB accident/incident data is cleaned and merged with volume data provided by the FAA website, which is then combined with information about the number of abbreviations, LIWC categories, Doc2Vec information, etc. provided by the ASRS dataset (that we analyzed before). When matching any two different sets of data, we utilize tracon\_code and the date of the event/summary. Anytime the tracon\_codes are not matched with any of the other datasets, we utilize NaNs to fill in the missing data.


## Running The Repository
First, you will need to upload, the ASRS dataset and place it in the folder `preprocess_asrs/datasets/`. We've excluded the file because of storage reasons (git LFS). After that, you simply need to run the following command.
```
./run_all
```

## Overall Pipeline
1. `preprocess_asrs/clean.py`: preprocesses the ASRS dataset by extracting tracon information. See `preprocess_asrs/README.md` for additional information.
2. `asrs_analysis/run_all`: analyzes ASRS dataset by categorizing all words used in the dataset. This also calculates cosine similarity metrics (using doc2vec) and liwc counts
3. `aviation_data_huiyi/run_all`: processes NTSB incident/accident data
    a. See `aviation_data_huiyi/README.md` for more information.
    b. This repository cleans the airport_codes and airport_names of the NTSB incident/accident data and it utilizes latitude/longitude informations to fill invalid rows.
4. `faa/clean_data.py`: processes FAA incident/accident data
    a. This code utilizes queries to iata.org to fill airport codes and airport names. It also cleans the names/codes.
5. `join_faa_ntsb.py`: joins results from previous two steps
6. `flight_vol.py`: combines FAA volume data with results from step 3
7. `combine.py`: combines result from step (4) with results of the abbreviation/LIWC/doc2vec analysis (2)

## Datasets
Before jumping into a detailed description of each step, here is some background information on the datasets involved.

### FAA/NTSB Accident/Incident Dataset
The subdirectories `aviation_data_huiyi` and `faa` deal with cleaning the NTSB and FAA accident/incident data respectively. Each dataset is a dataframe consisting of accidents and incidents occurring at various airports. The columns of interest include the tracon\_code/airport\_code, year and month. These datasets are transformed into a counts dataframe, where each row consists of airport\_code/tracon\_code, airport\_name year, month, and the number of incidents and accidents that occured in the FAA dataset and the NTSB dataset. Some example rows of the output are shown below (`airport_month_events.csv` for full csv).

| airport\_code | airport\_name | year | month | ntsb\_accidents | ntsb\_incidents | faa\_incidents | dataset |
| ------------- | ------------- | ---- | ----- | --------------- | --------------- | -------------- | ------- |
| ADY | Alldays Airport | 1991 | 7 | 1.0 | 0.0 | 0.0 | ntsb |
| NNB | Santa Ana Island | 1997 | 10 | 0.0 | 0.0 | 1.0 | faa |

The original datasets are saved in `aviation_data_huiyi/datasets/NTSB_AIDS_full.txt` and `faa/datasets/FAA_AIDS_full.txt`. The columns of interest for the NTSB dataset are `Investigation Type` (incident or accident), `Event Date`, `Airport Code`, and `Airport Name`, wherease in the FAA dataset they are `eventairport`, `eventtype` (incident or accident), and `localeventdate` (airport code is looked up via a script) found in the `faa/` subdirectory.

### FAA Volume Dataset
This dataset was queried from the following [link](https://aspm.faa.gov/opsnet/sys/tower.asp) and the data includes the number of flights from a given airport/facility at any year/month as well as the types of flights that occurred. These datasets are saved in `datasets/WEB-Report-*`. This data is not processed in any way and is simply joined with the rest of the data.

### ASRS Dataset
This dataset includes information on incidents/accidents that occurred and the reports that were written after each incident/accident occurred. These reports are saved in the `narrative` and `synopsis` fields. We create our own field `combined` which joins the two together. The original dataset is not provided (see above). However

## In-Depth Pipeline
### (1) Preprocessing ASRS Dataset
**Purpose**: extract tracon codes from the ASRS dataset (not provided) into new columns utilizing the `atcadvisory` column. The following section repeats information found in `preprocess_asrs/README.md`.

**Input**: all paths are from `preprocess_asrs/` as root.
* `datasets/ASRS 1988-2019.csv` full traffic control data (not provided)
* `datasets/wiki_code.csv` this is a dataframe scrapped from [wikipedia](https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_A)
* `datasets/tracon_code.csv`: this is a dataframe scrapped from [faa](https://www.faa.gov/about/office_org/headquarters_offices/ato/service_units/air_traffic_services/tracon/)

**Output**:
* `outputs/type_cts.png`: bar plot of most common types (not ATC) of atcadvisory column of original datasets
* `outputs/type_cts.csv`: dataframe of all types and how often they show up in atcadvisory column of original datasets
* `outputs/codes_cts.png`: bar plot of most common codes (not ATC) of atcadvisory column of original datasets
* `outputs/codes_cts.csv`: dataframe of all codes and how often they show up in atcadvisory column of original datasets
* `outputs/ASRS 1988-2019_extracted.csv`: new dataframe with added columns `{atc|info}_{code|type|repeated}(0-3)`. 

**Methodology**:  We are particularly interested in analyzing the atcadvisory column, which is typically formatted as follows: `Center ATL.Center; Tower ZZZ; UNICOM SFO`. Each row can therefore be separated into lists of `(code, type, repeated_type)` where `repeated_code` is T/F depending on whether or not the type is repeated after the code. Some codes are not ATC codes (ZZZ is not but ATL is), so we use the dataframe scraped from wikipedia and faa to determine which codes are ATC and which are miscellaneous info (and separate them).

**Running**:
```
cd preprocess_asrs/
python clean.py
cd ../
```

### (2) Analyzing ASRS Dataset
#### (2.a) abbrev\_words\_analysis.py
**Purpose**: Take the ASRS dataset and create a dataframe with known abbreviations (within narrative/synopsis/combined columns).

**Input**: 
* Main dataset: `ASRS 1988-2019_extracted.csv`
* Dictionaries: `FAA.csv`, `CASA.csv`, `IATA_IACO.csv`, `hand_code.csv`, `nasa_abbr.csv`, and `LIWC2015-dictionary-poster-unlocked.xlsx` 

**Output**:
* `total_cts_tagged_{narrative|synopsis|combined}.csv`: this is a dataframe where each row represents a word. Columns include a ct (number of occurences in dataset), a tag (categorization of the word), and the fullform version of the abbreviation (for each dictionary i.e iata_fullform, nasa_fullform, etc.). If a fullform does not exist, then it’s left blank
* A csv for each tag (pos_stopword, neg_nonword, etc. see future slides): subset of total\_cts\_tagged_{...}.
* `{unique_|}abrev_bar_plot_{narrative|synopsis|combined}.png`: bar plot of tag breakdown. If the first selection is ‘unique_’, then we only look at unique words (not total counts)

**Methodology**:
* We utilized dictionaries from CASA, FAA, NASA, and IATA (of known aviation abbreviations and their full-forms) as well as a hand-coded dictionary. All words that were found in the ASRS dataset and in the aviation dictionaries are marked immediately as abbreviations. However, some of these abbreviations were just common English words, so we utilized an English dictionary to filter out false-positives
* Overall approach
    * All words that are found in an aviation dictionary and not in the English dictionary are marked as abbreviations (pos_nonword: pos indicating that it was found in an aviation dictionary)
    * All words that are not found in aviation dictionary and not in the English dictionary are marked as abbreviations (neg_nonword)
    * We looked at the top words (in terms of counts) that are found in an aviation dictionary and in the English dictionary. Words that we thought were not common English words are marked as abbreviations (pos_handcoded_abrev). Note that this hand-coded dictionary is separate from the `hand_code.csv`
* We categorized all words that was found in the dataset regardless of whether or not they were marked as abbreviations
    * Neg_stopword: word that was not found in an aviation dictionary, and is a known English stopword (from nltk stopwords)
    * Neg_nonword: word that was not found in an aviation dictionary, nor the English dictionary
    * Neg_nonword_city_exception: a neg_nonword that was also part of a name of a US city
        * Ex.: ‘York’ from New York
    * Neg_nonalpha_abrev:  a neg_nonword with non-alphanumeric characters (these are not marked as abbreviations)
    * Pos_stopword: word that was found in an aviation dictionary and is an English stopword
    * Pos_nonword: word that was found in an aviation dictionary and is not an English word
    * Pos_word: word that was found in an aviation dictionary and is an English word
    * Pos_handcoded_abrev: description in previous slide
    * Pos_iata_only_words: word that was only found in the iata dictionary (marked as non-abrev)

**Running**:
```
cd asrs_analysis
python abbrev_words_analysis.py
cd ../
``` 
#### (2.b) top\_down.py
**Purpose**: Organize ASRS dataset via tracon_month and create a counts dataframe mapping tracon_month to the number of times pos_nonwords, overall abbreviations, and (faa|casa|iata|hand|nasa) abbreviations show up.

**Input**: 
* `ASRS 1988-2019_extracted.csv`: main dataset
* `total_cts_tagged_{narrative|synopsis|combined}.csv`: dataframe consisting of number of times each words show up as well as a categorization of each word (explained in prev slide)
* Dictionaries: `FAA.csv`, `CASA.csv`, `IATA_IACO.csv`, `hand_code.csv`, `nasa_abbr.csv`, and `LIWC2015-dictionary-poster-unlocked.xlsx` 

**Output**:
* `Tracon_month_{narrative|synopsis|combined}.csv`: dataframe consisting of tracon_months and their associated counts 
    * List of columns
        * `tracon_month` (this is the index)
        * `pos_nonword_{ct|unique_ct}`: the number of times the tag `pos_nonword` shows up in the tracon_month
            * If unique, the number of unique `pos_nonword`s that show up in the tracon_month
        * `all_abrevs_no_overcount_{ct|unique_ct}`: the number of times all abbreviations show up in the tracon_month.
        * `{casa|faa|iata|nasa|hand}_{ct|unique_ct}`: the number of times abbreviations show up for each given dictionary

 

<!-- **Files used** -->
<!-- 1. `d2v_tracon_month_{narrative|synopsis|combined}_{1|3|6|12|inf}mon.csv`: this is the result of creating doc2vec representations of the columns (narrative, synopsis, combined of ASRS dataset), and comparing these vectors to each other utilizing the cosine\_similarity metric. Each row represents a `tracon_month`: an airport_code or tracon_code paired with a year and month. The `{1|3|6|12|inf}` selection indicates over what time period is compared. For instance, if we are looking at a particular row in the ASRS dataset with a date of January 2011, and if the selection is 1 month, the doc2vec numbers are calculated over December 2010. If the selection is 3 months, then the doc2vec numbers are calculated over October - November of 2010. -->
<!--     * Column format: `d2v_{cos_sim|num_comp}{|_other_tracon|_all_tracon}{_replace|}` -->
<!--         1. `{cos_sim|num_comp}`: `cos_sim` means the number given is an average cosine_similarity metric over some qualifiers (see other selections) whereas `num_comp` is the number of comparisons made in the average cosine similarity metric. -->
<!--         2. `{|_other_tracon|_all_tracon}`: if this selection is blank, then the comparisons are made over the given tracon and year/month. For instance, if the given tracon/year/month is SFO/January/2011, then the cos_sim number is calculated to be the average cosine similarity of pairwise doc2vec vectors (calculated via the narrative/synopsis/combined column) during January 2011 of the SFO tracon/airport. If the selection is `_other_tracon`, then the comparisons are made during January 2011 to all other tracons except SFO. If the selection is `_all_tracon`, then the comparisons are made to all tracons during January 2011. -->
<!--  -->
<!--         3. `{_replace|}`: if the selection has `replace` then the doc2vec calculations occurred after replacing all abbreviations (found in this repository TO-DO) are replaced with their fullforms. If the selection is blank, then no such replacement is done. -->
<!--     * Examples: we will use the same example as above with the date of January 2011 in SFO. -->
<!--         1. `d2v_cos_sim_other_tracon`: this is the average cosine similarity calculated between pairwise comparisons of doc2vec vectors from the reports made in January 2011 in SFO to all reports made in January 2011 in tracons/airports outside of SFO. -->
<!--         2. `d2v_num_comp_all_tracon_replace`: this is the number of comparisons made between doc2vec vectors from the reports made in January 2011 in SFO to all reports made in Junary 2011. -->
<!-- 2. `liwc_tracon_month_{narrative|synopsis|combined}.csv`: LIWC is a categorization of some number of words (for instance common adverbs, family related words, swear words, etc.). This csv takes each `tracon_month` from the ASRS dataset and counts how many of each category was used during that `tracon_month`. -->
<!-- 3. `total_cts_tagged_{narrative|synopsis|combined}.csv`: this is the result of the abbreviation analysis done in the other repository on the ASRS dataset. This dataframe consists of a list of words/acronyms that appear in the corresponding column (narrative/synopsis/combined), the number of times they appear in the whole dataset, the corresponding full-forms, and whether or not we consider them to be abbreviations. -->
<!--     * This csv was created utilizing some aviation dictionaries (which have some common abbreviations and their full forms). The five dictionaries are CASA, FAA, IATA, NASA and HAND (or hand-coded) dictionaries. -->
<!--     * Each word is given a tag or categorization. The tag starting with the prefix `pos_` indicates that the word was found in any of the aviation dictionaries. Whereas if the tag starts with `neg_` then the word was not found in any of the aviation dictionaries. -->
<!--         1. `pos_word`: a word that was found in an aviation dictionary and is an English word according to the enchant dictionary -->
<!--         2. `pos_stopword`: a word that was found in an aviation dictionary and is an English stopword -->
<!--         3. `pos_nonword`: a word that was found in an aviation dictionary and is not an English word -->
<!--         4. `pos_iata_only_words`: a word that was found only in the iata dictionary -->
<!--         5. `pos_handcoded_abrev`: a word that was found in an aviation dictionary and is an English word. However, we mark these to be abbreviations by inspection (in other words they are `pos_word` that are actually abbreviations despite being in the English dictionary) -->
<!--         6. `neg_word`: a word that was not found in an aviation dictionary and is an English word. -->
<!--         7. `neg_stopword`: a word that was not found in an aviation dictionary and is an English stopword. -->
<!--         8. `neg_nonword`: a word that was not found in an aviation dictionary and is not an English word. -->
<!--         9. `neg_nonword_city_exception`: these are `neg_nonwords` that overlap with a city name (ex: york) -->


## Organization of Repository
1. `preprocess_asrs/` is the directory in which ASRS tracon codes are extracted and cleaned
2. `asrs_analysis/` is the directory responsible for analyzing the ASRS dataset via LIWC, Doc2Vec, and abbreviation analysis.
3. `abrev_datasets/` includes the results from the abbreviation/ASRS repository
4. `datasets/` includes the data from the FAA website.
5. `faa/` includes the repository that cleans the FAA incident/accident data
6. `aviation_data_huiyi/`: includes the repository that cleans the NTSB incident/accident data.
7. `join_faa_ntsb.py`: joins the results from `faa/` and `aviation_data_huiyi/` into one dataset. This creates the file `airport_month_events.csv`
8. `flight_vol.py`: combines `airport_month_events.csv` with the FAA volume data scrapped from the website above. This creates `combined_vol_incident.csv`
9. `combine.py`: this combines `combined_vol_incident.csv` with all the abbreviation datasets from the ASRS repository. This creates the files: `final_dataset_{1|3|6|12|inf}.csv`.

