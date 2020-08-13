# Aviation Data Consolidation
This repository cleans FAA/NTSB accident/incident data with volume data provided by the FAA website, which is then combined with information about the number of abbreviations, LIWC categories, Doc2Vec information, etc. provided by the ASRS dataset. When matching any two different sets of data, we utilize tracon\_code and the date of the event/summary. Anytime the tracon\_codes are not matched with any of the other datasets, we utilize NaNs to fill in the missing data.

## Datasets
### FAA/NTSB Accident/Incident Dataset
The subdirectories `aviation_data_huiyi` and `faa` deal with cleaning the NTSB and FAA accident/incident data respectively. Each dataset is a dataframe consisting of accidents and incidents occurring at various airports. The columns of interest include the tracon\_code/airport\_code, year and month. These datasets are transformed into a counts dataframe, where each row consists of airport\_code/tracon\_code, airport\_name year, month, and the number of incidents and accidents that occured in the FAA dataset as well as the NTSB dataset. Some example rows of the output are shown below (`airport_month_events.csv` for full csv).

| airport\_code | airport\_name | year | month | ntsb\_accidents | ntsb\_incidents | faa\_incidents | dataset |
| ------------- | ------------- | ---- | ----- | --------------- | --------------- | -------------- | ------- |
| ADY | Alldays Airport | 1991 | 7 | 1.0 | 0.0 | 0.0 | ntsb |
| NNB | Santa Ana Island | 1997 | 10 | 0.0 | 0.0 | 1.0 | faa |

The original datasets are saved in `aviation_data_huiyi/datasets/NTSB_AIDS_full.txt` and `faa/datasets/FAA_AIDS_full.txt`. The columns of interest for the NTSB dataset are `Investigation Type` (incident or accident), `Event Date`, `Airport Code`, and `Airport Name`, wherease in the FAA dataset they are `eventairport`, `eventtype` (incident or accident), and `localeventdate` (airport code is looked up via a script) found in the `faa/` subdirectory.

### FAA Volume Dataset
This dataset was queried from the following [link](https://aspm.faa.gov/opsnet/sys/tower.asp) and the data includes the number of flights from a given airport/facility at any year/month as well as the types of flights that occurred. These datasets are saved in `datasets/WEB-Report-*`. This data is not processed in any way and is simply joined with the rest of the data.

### ASRS Dataset
This dataset includes information on incidents/accidents that occurred and the reports that were written after each incident/accident occurred. These reports are saved in the `narrative` and `synopsis` fields. We create our own field `combined` which joins the two together. This dataset is processed in a different repository (TO-DO: link here), but the results of that repository are used here (via the csvs in `abrev_datasets/`). 

**Files used**
1. `d2v_tracon_month_{narrative|synopsis|combined}_{1|3|6|12|inf}mon.csv`: this is the result of creating doc2vec representations of the columns (narrative, synopsis, combined of ASRS dataset), and comparing these vectors to each other utilizing the cosine\_similarity metric. Each row represents a `tracon_month`: an airport_code or tracon_code paired with a year and month. The `{1|3|6|12|inf}` selection indicates over what time period is compared. For instance, if we are looking at a particular row in the ASRS dataset with a date of January 2011, and if the selection is 1 month, the doc2vec numbers are calculated over December 2010. If the selection is 3 months, then the doc2vec numbers are calculated over October - November of 2010.
    * Column format: `d2v_{cos_sim|num_comp}{|_other_tracon|_all_tracon}{_replace|}`
        1. `{cos_sim|num_comp}`: `cos_sim` means the number given is an average cosine_similarity metric over some qualifiers (see other selections) whereas `num_comp` is the number of comparisons made in the average cosine similarity metric.
        2. `{|_other_tracon|_all_tracon}`: if this selection is blank, then the comparisons are made over the given tracon and year/month. For instance, if the given tracon/year/month is SFO/January/2011, then the cos_sim number is calculated to be the average cosine similarity of pairwise doc2vec vectors (calculated via the narrative/synopsis/combined column) during January 2011 of the SFO tracon/airport. If the selection is `_other_tracon`, then the comparisons are made during January 2011 to all other tracons except SFO. If the selection is `_all_tracon`, then the comparisons are made to all tracons during January 2011.

        3. `{_replace|}`: if the selection has `replace` then the doc2vec calculations occurred after replacing all abbreviations (found in this repository TO-DO) are replaced with their fullforms. If the selection is blank, then no such replacement is done.
    * Examples: we will use the same example as above with the date of January 2011 in SFO.
        1. `d2v_cos_sim_other_tracon`: this is the average cosine similarity calculated between pairwise comparisons of doc2vec vectors from the reports made in January 2011 in SFO to all reports made in January 2011 in tracons/airports outside of SFO.
        2. `d2v_num_comp_all_tracon_replace`: this is the number of comparisons made between doc2vec vectors from the reports made in January 2011 in SFO to all reports made in Junary 2011.
2. `liwc_tracon_month_{narrative|synopsis|combined}.csv`: LIWC is a categorization of some number of words (for instance common adverbs, family related words, swear words, etc.). This csv takes each `tracon_month` from the ASRS dataset and counts how many of each category was used during that `tracon_month`.
3. `total_cts_tagged_{narrative|synopsis|combined}.csv`: this is the result of the abbreviation analysis done in the other repository on the ASRS dataset. This dataframe consists of a list of words/acronyms that appear in the corresponding column (narrative/synopsis/combined), the number of times they appear in the whole dataset, the corresponding full-forms, and whether or not we consider them to be abbreviations.
    * This csv was created utilizing some aviation dictionaries (which have some common abbreviations and their full forms). The five dictionaries are CASA, FAA, IATA, NASA and HAND (or hand-coded) dictionaries.
    * Each word is given a tag or categorization. The tag starting with the prefix `pos_` indicates that the word was found in any of the aviation dictionaries. Whereas if the tag starts with `neg_` then the word was not found in any of the aviation dictionaries.
        1. `pos_word`: a word that was found in an aviation dictionary and is an English word according to the enchant dictionary
        2. `pos_stopword`: a word that was found in an aviation dictionary and is an English stopword
        3. `pos_nonword`: a word that was found in an aviation dictionary and is not an English word
        4. `pos_iata_only_words`: a word that was found only in the iata dictionary
        5. `pos_handcoded_abrev`: a word that was found in an aviation dictionary and is an English word. However, we mark these to be abbreviations by inspection (in other words they are `pos_word` that are actually abbreviations despite being in the English dictionary)
        6. `neg_word`: a word that was not found in an aviation dictionary and is an English word.
        7. `neg_stopword`: a word that was not found in an aviation dictionary and is an English stopword.
        8. `neg_nonword`: a word that was not found in an aviation dictionary and is not an English word.
        9. `neg_nonword_city_exception`: these are `neg_nonwords` that overlap with a city name (ex: york)


## Organization of Repository
1. `abrev_datasets/` includes the results from the abbreviation/ASRS repository
2. `datasets/` includes the data from the FAA website.
3. `faa/` includes the repository that cleans the FAA incident/accident data
4. `aviation_data_huiyi/`: includes the repository that cleans the NTSB incident/accident data.
5. `join_faa_ntsb.py`: joins the results from `faa/` and `aviation_data_huiyi/` into one dataset. This creates the file `airport_month_events.csv`
6. `flight_vol.py`: combines `airport_month_events.csv` with the FAA volume data scrapped from the website above. This creates `combined_vol_incident.csv`
7. `combine.py`: this combines `combined_vol_incident.csv` with all the abbreviation datasets from the ASRS repository. This creates the files: `final_dataset_{1|3|6|12|inf}.csv`.

## Pipeline
The assumption in this repository is that the ASRS/other repository has already been run and their results are included in the `abrev_datasets` subdirectory. Then the following pipeline occurs:
1. `aviation_data_huiyi/run_all`: processes NTSB incident/accident data
    a. See `aviation_data_huiyi/README.md` for more information.
    b. This repository cleans the airport_codes and airport_names of the NTSB incident/accident data and it utilizes latitude/longitude informations to fill invalid rows.
2. `faa/clean_data.py`: processes FAA incident/accident data
    a. This code utilizes queries to iata.org to fill airport codes and airport names. It also cleans the names/codes.
3. `join_faa_ntsb.py`: joins results from previous two steps
4. `flight_vol.py`: combines FAA volume data with results from step 3
5. `combine.py`: combines result from step (4) with results of the abbreviation repo (TODO link)

## Running The Repository
```
./run_all
```
