import pandas as pd
import numpy as np
import os
import chardet
import kagglehub

def load_dataset(dataset_name="vivek468/superstore-dataset-final",
                 file_name="Sample - Superstore.csv"):
    path = kagglehub.dataset_download(dataset_name)
    csv_file = os.path.join(path, file_name)

    with open(csv_file, "rb") as f:
        enc = chardet.detect(f.read())
    print("Encodage détecté:", enc)

    df = pd.read_csv(csv_file, encoding=enc['encoding'])
    return df

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(r'[^\w\s]', '', regex=True)
                  .str.replace(r'\s+', '_', regex=True)
    )
    if "row_id" in df.columns:
        df = df.drop(columns=["row_id"])
    return df

def convert_types(df):
    df = df.copy()
    df['postal_code'] = df['postal_code'].astype(str)
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
    df['sales'] = df['sales'].astype(float)
    df['discount'] = df['discount'].astype(float)
    df['profit'] = df['profit'].astype(float)
    return df

def pipeline(df):
    df = clean_columns(df)
    df = convert_types(df)
    df = df.drop_duplicates()
    # df['shipping_time_days'] = (df['ship_date'] - df['order_date']).dt.days
    return df

def get_clean_data(save_csv=True):
    df_raw = load_dataset()
    df_clean = pipeline(df_raw)
    if save_csv:
        os.makedirs("../data", exist_ok=True)
        df_clean.to_csv("../data/clean_superstore.csv", index=False)
        print("✅ CSVs sauvegardés dans ../data/")
    return df_clean
