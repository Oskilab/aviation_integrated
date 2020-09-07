ifr_vfr_dict = {
    'itinerant': 'itnr',
    'general': 'gen',
    'overflight': 'ovrflt'
}

def rename_cols(pd):
    rename_dict = {}
    for col in pd.columns:
        if 'IFR' in col or 'VFR' in col:
            split_col = col.lower().split("\t")
            start_str = ''.join([ifr_vfr_dict.get(x, x) for x in split_col[1].split()])
            end_str = '_'.join([ifr_vfr_dict.get(x, x) for x in split_col[0].split()])
            rename_dict[col] = f'{start_str}_{end_str}'
        elif 'Local' in col:
            split_col = col.lower().split()
            rename_dict[col] = f'{split_col[1]}_{split_col[0]}'
    return pd.rename(rename_dict, axis = 1)
