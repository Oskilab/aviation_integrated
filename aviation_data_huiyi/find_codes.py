from collections import namedtuple
from tqdm import tqdm
from urllib.parse import quote
from urllib.error import HTTPError
import requests, pandas as pd, urllib.request as request, pickle
page_google = True
key = 'AIzaSyCR9FRYW-Y7JJbo4hU682rn6kJaUA5ABUc'

coverage = namedtuple('coverage', ['part', 'total'])

# preprocess data
full = pd.read_csv('datasets/NTSB_AviationData_new.txt', sep = "|")
strip_cols = [' Airport Code ' , ' Airport Name ', ' Latitude ', ' Longitude ']
for col in strip_cols:
    full[col] = full[col].str.strip()
full[' Airport Name '] = full[' Airport Name '].str.replace("N/A", "")

if page_google:
    empty_code_sel = full[' Airport Code '] == ''
    empty_lat = full[' Latitude '] == ''
    empty_lon = full[' Longitude '] == ''
    unique_lat_lon = full.loc[(~(empty_lat | empty_lon)) & empty_code_sel, \
                                  [" Latitude ", " Longitude "]].copy().drop_duplicates()
    tqdm_obj =  tqdm(unique_lat_lon.iterrows(), total = unique_lat_lon.shape[0], desc = "found 0")
    errors = {}
    lat_lon_records = []
    for idx, row in tqdm_obj:
        lat, lon = row
        lat, lon = float(lat), float(lon)
        map_api_str = f'https://maps.googleapis.com/maps/api/place/nearbysearch/json?key={key}' + \
            f'&location={quote(str(lat))},{quote(str(lon))}&rankby=distance&type=airport'

        try:
            response = requests.get(map_api_str.replace(" ",""))
            response.raise_for_status()
            
            # access JSON content
            json = response.json()
            if json['status'] != 'OK':
                errors[idx] = json['status']
            else:
                all_res = []
                for idx, results in enumerate(json['results']):
                    res_dict = {'lat': lat, 'lon': lon}
                    res_dict['airport_name'] = results['name']
                    res_dict['airport_lat'] = results['geometry']['location']['lat']
                    res_dict['airport_lon'] = results['geometry']['location']['lng']
                    res_dict['vicinity'] = results['vicinity']
                    all_res.append(pd.Series(res_dict).add_suffix(f"_{idx}"))
                res = pd.concat(all_res, axis = 0)
                lat_lon_records.append(res)
                tqdm_obj.set_description(f"found {len(lat_lon_records)}")

                # for good measure
                if (len(lat_lon_records) % 50) == 0:
                    pd.DataFrame.from_records(lat_lon_records).to_csv('results/lat_lon_records.csv')
                    pickle.dump(errors, open('results/idx_errors.pckl', 'wb'))
        except HTTPError as http_err:
            errors[idx] = str(http_err)
        except Exception as err:
            errors[idx] = str(err)
    lat_lon_pd = pd.DataFrame.from_records(lat_lon_records)
    lat_lon_pd.to_csv('results/lat_lon_records.csv')
    pickle.dump(errors, open('results/idx_errors.pckl', 'wb'))
else:
    lat_lon_pd = pd.read_csv('results/lat_lon_records.csv')
