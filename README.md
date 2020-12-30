# Aviation Integrated

This repository processes the ASRS dataset by extracting the tracon information. Then, the ASRS columns `narrative, synopsis, and combined` (which are descriptions and summaries of accidents/incidents that we have access to) are all analyzed utilizing LIWC, Doc2Vec, and word categorization. Then, the FAA/NTSB accident/incident data is cleaned and merged with volume data provided by the FAA website, which is then combined with information about the number of abbreviations, LIWC categories, Doc2Vec information, etc. provided by the ASRS dataset (that we analyzed before). When matching any two different sets of data, we utilize tracon\_code and the date of the event/summary. Anytime the tracon\_codes are not matched with any of the other datasets, we utilize NaNs to fill in the missing data.


## Running The Repository
First, you will need to upload, the ASRS dataset and place it in the folder `preprocess_asrs/datasets/`. We've excluded the file because of storage reasons (git LFS). Furthermore, you will need to add the files `faa_ntsb_analysis/datasets/NTSB_AviationData_new.txt`,`faa_ntsb_analysis/datasets/FAA_AIDS_full.txt`, `faa_ntsb_analysis/FAA_AIDS_addition.csv`.

After that, you simply need to run the following command.
```
bash -e ./run_all
```
If you are running this locally, then run:
```
bash -e ./run_all -t
```
`-t` stands for test. This will only select a subset of the data to analyze in the most computationally expensive parts of the pipeline

## A Note about Handcoded Files
This is a list of handcoded files that are a part of the pipeline (that may need to be modified in the future

1. `asrs_analysis/dictionaries/combined_neg_nonword_handcode2.csv`: dataframe consisting of words we've categorized as `neg_nonword`, or words that are not found in any of the aviation dictionaries, and are also not considered to be english words via the enchant library. As a rule, we've ruled these to be abbreviations, but we've hand-coded some portion of them to either keep them as abbreviations or to remove them (see the file for the format)
2. `faa_ntsb_analysis/datasets/FAA_key.xlsx`: dataframe that maps `event_airport` (of FAA incident/accident dataset) to its corresponding tracon\_key
3. `faa_ntsb_analysis/datasets/NTSB_airportcode_fix.xlsx`: this is a dataframe that maps an uncleaned airport code like `KSFO` to a cleaned airport code like `SFO`. Must have the columns `fixed_airport_code`, and `airport_code` in that order.
4. `faa_ntsb_analysis/datasets/not_matched_full_v1.csv`: this is handcoded file that maps from FAA airport name to tracon\_code


Requirements for each file
1. `asrs_analysis/dictionaries/combined_neg_nonword_handcode2.csv`: must have the columns: `add_to_realworld_dictionary` `add_to_airport` `mispelled_word_fix`.
2. `faa_ntsb_analysis/datasets/FAA_key.xlsx`: must have `tracon_key` and `event_airport` columns in that order
3. `faa_ntsb_analysis/datasets/NTSB_airportcode_fix.xlsx`: must have the columns `fixed_airport_code`, and `airport_code` in that order.


## Overall Pipeline
1. `faa_ntsb_analysis/`: this processes the incident/accident data provided by the FAA/NTSB datasets. There are some issues (faa has no airport code, but has airport names, and ntsb has missing airport codes) that this repository aims to fix via web requests
    a. Pipeline:
        1. `python find_faa_code.py`: this finds the airport codes corresponding to each incident/accident row, and organizes the data into tracon\_months
        2. `python find_ntsb_code.py`: this does the same as above for the NTSB dataset
2. `join_faa_ntsb.py`: joins results from step 1
3. `preprocess_asrs/clean.py`: preprocesses the ASRS dataset by extracting tracon information. See `preprocess_asrs/README.md` for additional information.
4. `asrs_analysis/run_all`: analyzes ASRS dataset by categorizing all words used in the dataset. This also calculates cosine similarity metrics (using doc2vec) and liwc counts
5. `flight_vol.py`: combines FAA volume data with results from step 2
6. `combine.py`: combines result from step (4) with results of the abbreviation/LIWC/doc2vec analysis (2)

## Organization of Repository
1. `preprocess_asrs/` is the directory in which ASRS tracon codes are extracted and cleaned
2. `asrs_analysis/` is the directory responsible for analyzing the ASRS dataset via LIWC, Doc2Vec, and abbreviation analysis.
3. `datasets/` includes the data from the FAA website (volume data).
4. `faa_ntsb_analysis/` includes the repository that cleans the FAA/NTSB incident/accident data.
6. `join_faa_ntsb.py`: joins the results from `faa_ntsb_analysis/` into one dataset. This creates the file `airport_month_events.csv`
7. `flight_vol.py`: combines `airport_month_events.csv` with the FAA volume data scrapped from the website above. This creates `combined_vol_incident.csv`
8. `combine.py`: this combines `combined_vol_incident.csv` with all the abbreviation datasets from the ASRS repository. This creates the files: `results/final/final_dataset_{col}_{1|3|6|12}.csv` and `results/final_dataset.csv`

## Datasets
Before jumping into a detailed description of each step, here is some background information on the datasets involved.

### FAA/NTSB Accident/Incident Dataset
The subdirectory `faa_ntsb_analysis` deal with cleaning the NTSB and FAA accident/incident data. Each dataset (of FAA/NTSB) is a dataframe consisting of accidents and incidents occurring at various airports. The columns of interest include the tracon\_code/airport\_code, year and month. These datasets are transformed into a counts dataframe, where each row consists of airport\_code/tracon\_code, airport\_name year, month, and the number of incidents and accidents that occured in the FAA dataset and the NTSB dataset. Some example rows of the output are shown below (`airport_month_events.csv` for full csv).

| airport\_code | airport\_name | year | month | ntsb\_accidents | ntsb\_incidents | faa\_incidents | dataset | NTSB\_FAA\_incidents\_total | NTSB\_FAA\_incidents\_total\_nodups |
| ------------- | ------------- | ---- | ----- | --------------- | --------------- | -------------- | ------- | --------------------------- | ----------------------------------- | 
| ADY | Alldays Airport | 1991 | 7 | 1.0 | 0.0 | 0.0 | ntsb | 1.0 | 1.0 |
| NNB | Santa Ana Island | 1997 | 10 | 0.0 | 0.0 | 1.0 | faa | 1.0 | 1.0 |

The original datasets are saved in `faa_ntsb_analysis/datasets/NTSB_AviationData_new.txt`,`faa_ntsb_analysis/datasets/FAA_AIDS_full.txt`, `faa_ntsb_analysis/FAA_AIDS_addition.csv`. The columns of interest for the NTSB dataset are `Investigation Type` (incident or accident), `Event Date`, `Airport Code`, and `Airport Name`, wherease in the FAA dataset they are `eventairport`, `eventtype` (incident or accident), and `localeventdate` (airport code is looked up via a script) found in the `faa/` subdirectory.

### FAA Volume Dataset
This dataset was queried from the following [link](https://aspm.faa.gov/opsnet/sys/tower.asp) and the data includes the number of flights from a given airport/facility at any year/month as well as the types of flights that occurred. These datasets are saved in `datasets/WEB-Report-*`. This data is not processed in any way and is simply joined with the rest of the data.

### ASRS Dataset
This dataset includes information on incidents/accidents that occurred and the reports that were written after each incident/accident occurred. These reports are saved in the `narrative` and `synopsis` fields. We create our own field `combined` which joins the two together. The original dataset is not provided (see above). However

## In-Depth Pipeline
### (1) Cleaning NTSB/FAA Data (faa\_ntsb\_analysis/)
**Purpose**: take the FAA/NTSB incident/accident data and fill in the tracon\_code information (for the rows that are missing) as well as save some backup files for potential use in the future. Most importantly, this also creates a dataframe of tracon_months and the number of ntsb/faa incidents and accidents that occured within that tracon_month

**Input**:
* `NTSB_AviationData_new.txt`: all incidents/accident data from NTSB dataset 
* `FAA_AIDS_full.csv`: the main FAA incident/accident dataset
* `FAA_AIDS_addition.csv`: the second portion of FAA incident/accident dataset
* `worldcities.csv`: list of worldcities and their corresponding latitudes/longitudes. This was downloaded from this [website](https://simplemaps.com/data/world-cities)
* `us_state_abbrev.csv`: dataframe from state abbreviation to full capitalized (only first letter) state name

**Output**:
* Main outputs:
    * `NTSB_AIDS_full_processed.csv`: for each tracon_month, calculate the number of ntsb_incidents and ntsb_accidents
    * `FAA_AIDS_full_processed.csv`: the faa_incidents organized via tracon_month.
* Ancillary outputs:
    * `backup_ntsb.csv`: latitude/longitudes mapping to nearest airports to latitude/longitudes (utilizing world-airport-codes website). Currently not used, but may be used in the future. Note that only some rows have latitude/longitude data
    * `discarded_ntsb.csv`: rows of NTSB dataset that are discarded. Currently we are only utilizing the rows with non-empty airport name or airport code
    * `tracon_date_ntsb.csv`: dataframe of day, month, year, and tracon\_code, and the number of times that combination shows up in the NTSB dataset. This is used later to deal with overlapping results b/w NTSB and FAA.
    * `backup_faa.csv`: if the airportname is nan, then we look at city names. We combine this with a list of downloaded worldcities (and their corresponding latitude/longitude), to create a list of potential airports
    * `tracon_date_faa.csv`: dataframe of day, month, year, and tracon\_code, and the number of times that combination shows up in the FAA dataset. This is used later to deal with overlapping results b/w NTSB and FAA.

**Methodology**:
The NTSB dataset already has the field ` Airport Code ` inside the dataset, so weutilize this field (and perform some cleaning). Any analysis regarding querying wikipedia or latitude/longitude information is no longer used. We also utilize handcoded dictionaries (discussed above) to clean some ` Airport Code ` fields. The FAA dataset we also utilize a handcoded dictionary but utilize it to convert `eventairport` to a tracon code. Some others eventairport names are hand-coded via `faa_ntsb_analysis/datasets/not_matched_full_v1.csv`.

**Running**:
```
cd faa_ntsb_analysis
python find_faa_code.py
python find_ntsb_code.py
cd ../
```
### (2) Joining the FAA/NTSB datasets (join\_faa\_ntsb.py)
**Purpose**: merges the two datasets created by the previous two steps (`aviation_data_huiyi/` and `faa/` subdirectories)

**Input**:
* `NTSB_AIDS_full_processed.csv`: result from previous step
* `FAA_AIDS_full_processed`: result from previous step
**Output**:
* `Airport_month_events.csv`: organized by tracon_month. Each row has the number of faa_incidents, ntsb_accidents, ntsb_incidents, tracon_month, and a `dataset` field indicating which dataset the row came from.

**Running**:
```
python join_faa_ntsb.py
```

### (3) Preprocessing ASRS Dataset
**Purpose**: extract tracon codes from the ASRS dataset (not provided) into new columns utilizing the `atcadvisory` column. The following section repeats information found in `preprocess_asrs/README.md`.

**Input**: all paths are from `preprocess_asrs/` as root.
* `datasets/ASRS Data/*.csv` full traffic control data (not provided)
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

### (4) Analyzing ASRS Dataset (asrs\_analysis/)
#### (4.a) abbrev\_words\_analysis.py
**Purpose**: Take the ASRS dataset and create a dataframe with known abbreviations (within narrative/synopsis/combined columns). More specifically, we make a dataframe of all known words in the ASRS dataset, then we group these words into specific categories (discussed in detail below)

**Input**: 
* Main dataset: `ASRS 1988-2019_extracted.csv`
* Dictionaries: `FAA.csv`, `CASA.csv`, `IATA_IACO.csv`, `hand_code.csv`, `nasa_abbr.csv`, and `LIWC2015-dictionary-poster-unlocked.xlsx` 

**Output**:
* `total_cts_tagged_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv`: this is a dataframe where each row represents a word. Columns include a ct (number of occurences in dataset), a tag (categorization of the word), the fullform version of the abbreviation (for each dictionary i.e iata\_fullform, nasa\_fullform, etc.) and whether or not we consider the word to be an abbreviation. If a fullform does not exist, then it’s left blank
* A csv for each tag (pos\_stopword, neg\_nonword, etc. see future slides): subset of total\_cts\_tagged_{...}.
* `{unique_|}abrev\_bar\_plot_{narrative|synopsis|combined|callback|narrative_synopsis\_combined}.png`: bar plot of tag breakdown. If the first selection is ‘unique_’, then we only look at unique words (not total counts)

**Methodology**:
* We utilized dictionaries from CASA, FAA, NASA, and IATA (of known aviation abbreviations and their full-forms) as well as a hand-coded dictionary. All words that were found in the ASRS dataset and in the aviation dictionaries are marked immediately as abbreviations. However, some of these abbreviations were just common English words, so we utilized an English dictionary to filter out false-positives
* Overall approach
    * All words that are found in an aviation dictionary and not in the English dictionary are marked as abbreviations (pos_nonword: pos indicating that it was found in an aviation dictionary)
    * All words that are not found in aviation dictionary and not in the English dictionary are marked as abbreviations (neg_nonword)
    * We looked at the top words (in terms of counts) that are found in an aviation dictionary and in the English dictionary. Words that we thought were not common English words are marked as abbreviations (pos_handcoded_abrev). Note that this hand-coded dictionary is separate from the `hand\_code.csv`
* We categorized all words that was found in the dataset regardless of whether or not they were marked as abbreviations
    * Neg_stopword: word that was not found in an aviation dictionary, and is a known English stopword (from nltk stopwords)
    * Neg_nonword: word that was not found in an aviation dictionary, nor the English dictionary
    * Neg_word_hand2: neg-nonwords that we manually coded to be negwords again (this is done through a handcoded dataframe (see above)
    * Neg_airport: neg-nonwords that turned out to be airports (see handcoded dataframe)
    * Neg_nonword_city_exception: a neg_nonword that was also part of a name of a US city
        * Ex.: ‘York’ from New York
    * Neg_nonalpha_abrev:  a neg_nonword with non-alphanumeric characters (these are not marked as abbreviations)
    * Pos_stopword: word that was found in an aviation dictionary and is an English stopword
    * Pos_nonword: word that was found in an aviation dictionary and is not an English word
        * `pos_nwrd_ct = FAA + CASA + IATA + NASA + handcoded - english_words`
    * Pos_word: word that was found in an aviation dictionary and is an English word
    * Pos_handcoded_abrev: description in previous slide
    * Pos_iata_only_words: word that was only found in the iata dictionary (marked as non-abrev)

**Running**:
```
cd asrs_analysis
python abbrev_words_analysis.py
cd ../
``` 
#### (4.b) top\_down.py
**Purpose**: Organize ASRS dataset via tracon_month and create a counts dataframe mapping tracon_month to the number of times pos_nonwords, overall abbreviations, and (faa|casa|iata|hand|nasa) abbreviations show up.

**Input**: 
* `ASRS 1988-2019_extracted.csv`: main dataset
* `total_cts_tagged_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv`: dataframe consisting of number of times each words show up as well as a categorization of each word (explained in prev section)
* Dictionaries: `FAA.csv`, `CASA.csv`, `IATA_IACO.csv`, `hand_code.csv`, `nasa_abbr.csv`, and `LIWC2015-dictionary-poster-unlocked.xlsx` 

**Output**:
* `tracon_month_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv`: dataframe consisting of tracon_months and their associated counts 
    * List of columns
        * `tracon_month` (this is the index)
        * `pos_nwrd_{ct|unique_ct}`: the number of times the tag `pos_nonword` shows up in the tracon_month
            * If unique, the number of unique `pos_nonword`s that show up in the tracon_month
            * `pos_nwrd_ct = FAA + CASA + IATA + NASA + handcoded - english_words`: note that if a word occurs in multiple dictionaries (FAA and CASA for example), it is not counted twice.
        * `abrvs_no_ovrcnt_{ct|unique_ct}`: the number of times all abbreviations show up in the tracon_month.
            * `abrvs_no_ovrcnt = pos_nwrd + pos_handcoded_abrev + neg_nonword` (any word that is counted as an abbreviation), as the name implies we do not overcount if a word occurs in multiple categories
        * `{casa|faa|iata|nasa|hand}\_{ct|unq\_ct}{|\_all|\_out}`: the number of times abbreviations show up for each given dictionary
            * The {|\_all|\_out} selection determines whether or not we are calculating counts within a specific tracon (for a year/month time period), for all tracons within a year/month time period, or for all tracons outside our specific tracon within a year/month time period (respectively).
        * `{narr|syn|narrsyn|all}\_{avg_wc|wc}`: number of words in given tracon_month for that particular column (narrative/synopsis/etc). avg\_wc refers to average word count per observation within the tracon\_month
        * `{narr|syn|narrsyn|all}\_wc_{out|all|prop}`: if the selection is out, then it's the word count outside the given tracon but within that year/month time period. If the selection is all, then it's the word count for all tracons within the year/month time period. If the selection is prop, then the field is the proportion of word counts within the given tracon\_month compared to all word counts within the same time period

**Running**:
```
cd asrs_analysis
python top_down.py
cd ../
``` 
 
#### (4.c) Doc2Vec Cosine Similarity (cos_sim.py)
**Purpose**: convert the narrative/synopsis/combined fields in the ASRS dataset into doc2vec vectors and calculate the average cosine similarity between vectors in a given tracon_month to other groups (explained below).

**Input**:
* `ASRS 1988-2019_extracted.csv`: Main dataset
* `total_cts_tagged_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv` from abbrev analysis

**Output**:
* `d2v_tracon_month_{narrative|synopsis|combined|callback|narrative_synopsis_combined}_{1|3|6|12|inf}mon.csv`: this is the result of creating doc2vec representations of the columns (narrative, synopsis, combined of ASRS dataset), and comparing these vectors to each other utilizing the cosine\_similarity metric. Each row represents a `tracon_month`: an airport_code or tracon_code paired with a year and month. The `{1|3|6|12|inf}` selection indicates over what time period is compared. For instance, if we are looking at a particular row in the ASRS dataset with a date of January 2011, and if the selection is 1 month, the doc2vec numbers are calculated over December 2010. If the selection is 3 months, then the doc2vec numbers are calculated over October - November of 2010.
    * Column format: `trcn_{|outinvout|all|invall}{|flfrm}{|ct}_{narr|syn|all|narrsyn}_{1|3|6|12|a}m`
        * `{|out|invout|all|invall}`: no selection means that the field is calculated over given tracon\_month (January 2011, SFO)
            * out: cosine similarity comparisons are between vectors within the same time period (but not within the given tracon, SFO)
            * invout: cosine similarity comparisons are done between vectors within the tracon\_month and vectors within the same time period (January 2011) that come from a different tracon (not SFO)
            * all: cosine similarity comparisons are between all vectors within the same time period (i.e January 2011)
            * invall: cosine similarity comparisons are done between vectors within the tracon\_month (Jan. 2011 SFO) and all vectors within the given time period (January 2011)
        * `{|flfrm}`: no selection indicates that abbreviations are not expanded and flfrm indicates that the abbreviations are replaced by their full form
        * `{|ct}`: no selection means that the given field is the average cosine similarity number, whereas ct indicates that the field is the number of comparisons made
        * `{narr|syn|all|narrsyn}`: which field is used
        * `{1|3|6|12|a}`: the time period used. a indicates all time and the numbers indicate number of months
    * Examples: we will use the same example as above with the date of January 2011 in SFO.
        * `trcn_invout_narr_1m`: this is the average cosine similarity calculated between pairwise comparisons of doc2vec vectors from the reports made in January 2011 in SFO to all reports made in January 2011 in tracons/airports outside of SFO (using the narrative column).
        * `trcn_invout_flfrm_ct`: this is the number of comparisons made between doc2vec vectors from the reports made in January 2011 in SFO to all reports made in Junary 2011.

**Running**:
```
cd asrs_analysis
python cos_sim.py
cd ../
``` 
#### (4.d) LIWC Analysis (liwc\_analysis.py)
**Purpose**: for each tracon\_month, calculate the counts of each LIWC category

**Input**:
* `ASRS 1988-2019_extracted.csv`: Main dataset
* `LIWC2015-dictionary-poster-unlocked.xlsx`: LIWC dictionary

**Output**:
`liwc_tracon_month_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv`: LIWC is a categorization of some number of words (for instance common adverbs, family related words, swear words, etc.). This csv takes each `tracon_month` from the ASRS dataset and counts how many of each category was used during that `tracon_month`.

**Running**:
```
cd asrs_analysis
python liwc_analysis.py
cd ../
```


### (5) Merging with Volume Data (flight\_vol.py)
**Purpose**: combine faa/ntsb accident/incident data from the volume data from this [link](https://aspm.faa.gov/opsnet/sys/tower.asp). Merged on same tracon_month.

**Input**:
* `airport_month_events.csv` from the result of `join_faa_ntsb.py` 
* `WEB-Report-xxxx.{csv|xls}`: these are the datasets queried from the link above.

**Output**:
* `Combined_vol_incident.csv`: airport_month_events.csv + volume data. 
    * They are combined on tracon_month. If some data cannot be matched, then they are filled with nas
* `nf_dates.csv`: stands for not found dates. If the date in `airport_month_events.csv` cannot be found in the vol data, but the code exists in the vol data, the row is added to `nf_dates.csv`
* `nf_codes.csv`: stands for not found codes. If the code doesn’t exist in the vol data, the row is added to `nf_codes.csv`

**Running**:
```
python flight_vol.py
```

### (6) Joining All Datasets (combine.py)
**Purpose**: combine asrs dataset, d2v dataset, liwc dataset, faa/ntsb incident/accident, volume data all together

**Input**:
* `tracon_month_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv`: processed ASRS dataset
* `combined_vol_data.py`: combined faa/ntsb incident/accident data with volume data
* `liwc_tracon_month_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv`: dataframe of tracon_months and their corresponding liwc category counts
* `d2v_tracon_month_{narrative|synopsis|combined|callback|narrative_synopsis_combined}_{1|3|6|12|inf}mon.csv`: doc2vec cosine similarity output

**Output**:
* `final_dataset.csv`: combined dataset with all relevant information. For full description of all the columns, see the google drive dictionary.

**Methodology**:
* We take each row from `combined_vol_incident.csv` and combine the corresponding d2v row, liwc row, and row from `tracon_month_{narrative|synopsis|combined|callback|narrative_synopsis_combined}.csv` (processed ASRS dataset). They’re combined starting on the month before the given month of `combined_vol_incident.csv` (see below)
    * Let’s say that the row from `combined_vol_incident.csv` has a tracon_month of SFO on January 2011
    * If the month\_range is 1, then the rows from LIWC, D2V, and processed ASRS dataset that has a tracon_month of SFO on December 2010 are joined with that of SFO on January 2011
    * If the month range is 12, then the rows from LIWC, D2V, and processed ASRS dataset has a tracon_month of SFO from January 2010 to December 2010

**Running**:
```
python combine.py
```


