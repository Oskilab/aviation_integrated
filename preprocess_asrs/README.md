# Requirements
* Pandas
* Numpy
* Matplotlib
* Python

# File Descriptions
* `datasets/ASRS 1988-2019.csv` full traffic control data
* `datasets/wiki_code.csv` this is a dataframe scrapped from [wikipedia](https://en.wikipedia.org/wiki/List_of_airports_by_IATA_airport_code:_A)
* `datasets/tracon_code.csv`: this is a dataframe scrapped from [faa](https://www.faa.gov/about/office_org/headquarters_offices/ato/service_units/air_traffic_services/tracon/)
* `outputs/type_cts.png`: bar plot of most common types (not ATC) of atcadvisory column of original datasets
* `outputs/type_cts.csv`: dataframe of all types and how often they show up in atcadvisory column of original datasets
* `outputs/codes_cts.png`: bar plot of most common codes (not ATC) of atcadvisory column of original datasets
* `outputs/codes_cts.csv`: dataframe of all codes and how often they show up in atcadvisory column of original datasets
* `outputs/ASRS 1988-2019_extracted.csv`: new dataframe with added columns `(atc|info)_(code|type|repeated)(0-3)`. This is explained below.

# Running
```
python clean.py 
```
This cleans `datasets/ASRS 1988-2019.csv`. We are particularly interested in analyzing the atcadvisory column, which is typically formatted as follows: `Center ATL.Center; Tower ZZZ; UNICOM SFO`. Each row can therefore be separated into lists of `(code, type, repeated_type)` where `repeated_code` is T/F depending on whether or not the type is repeated after the code. Some codes are not ATC codes (ZZZ is not but ATL is), so we use the dataframe scraped from wikipedia and faa to determine which codes are ATC and which are miscellaneous info (and separate them).

The final output has the added columns `(atc|info)_(code|type|repeated)(0-3)`, where atc/info determines what type of code is found, code/type/repeated is explained above and the digit represents the index. For instance, `atc_code0` represents the first atc code found in the column `atcadvisory` whereas `info_type2` represents the 3rd type found (that's not ATC related).
