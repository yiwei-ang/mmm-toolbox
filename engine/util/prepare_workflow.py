import json
import pandas as pd


def prepare_stan_data(data_df: pd.DataFrame):
    stan_data = {
        'N': int(len(data_df)),
    }
    for col in data_df.columns:
        if col in ['date']:
            continue
        else:
            stan_data[col] = data_df[col].tolist()

    with open('stan_data.json', 'w') as f:
        json.dump(stan_data, f)