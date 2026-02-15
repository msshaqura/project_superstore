import pandas as pd
import numpy as np
import os
import chardet
import kagglehub

# ---------------------------------------
# ---- 1️⃣ Téléchargement les données de KaggleHub ----
# ---------------------------------------

def load_dataset(dataset_name="vivek468/superstore-dataset-final", file_name="Sample - Superstore.csv"):
    """
    Télécharge le dataset depuis KaggleHub et retourne un DataFrame pandas.
    """
    # Télécharger le dataset
    path = kagglehub.dataset_download(dataset_name)
    csv_file = os.path.join(path, file_name)

    # Détection de l'encodage
    with open(csv_file, "rb") as f:
        raw = f.read()
    enc = chardet.detect(raw)
    print("Encodage détecté:", enc)

    # Lire le fichier CSV avec l'encodage correct
    df = pd.read_csv(csv_file, encoding=enc['encoding'])

    return df

# ---------------------------------------
# ---- 2️⃣ Nettoyage des colonnes ----
# ---------------------------------------

def clean_columns(df):
    """
    Nettoyage des noms de colonnes: minuscules, underscores, suppression de row_id
    """
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

# ---------------------------------------
# ---- 3️⃣ Conversion des types ----
# ---------------------------------------

def convert_types(df):
    """
    Conversion des colonnes aux types appropriés
    """
    df = df.copy()
    
    # Conversion en string
    df['postal_code'] = df['postal_code'].astype(str)
    
    # Conversion en datetime
    df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')
    df['ship_date'] = pd.to_datetime(df['ship_date'], errors='coerce')
    
    # Conversion en float
    df['sales'] = df['sales'].astype(float)
    df['discount'] = df['discount'].astype(float)
    
    return df

# ---------------------------------------
# ---- 4️⃣ Pipeline complet ----
# ---------------------------------------

def pipeline(df):
    """
    Pipeline complet: nettoyage colonnes, conversion types, suppression doublons
    """
    df = clean_columns(df)
    df = convert_types(df)
    
    # Vérification et suppression des doublons
    print("Nombre de doublons avant suppression:", df.duplicated().sum())
    df = df.drop_duplicates()
    print("Nombre de doublons après suppression:", df.duplicated().sum())
    
    # Création de la colonne shipping_time en jours
    df['shipping_time_days'] = (df["ship_date"] - df['order_date']).dt.days
    
    return df

# ---------------------------------------
# ---- 5️⃣ Fonction finale pour exporter ----
# ---------------------------------------

def get_clean_data(dataset_name="vivek468/superstore-dataset-final", file_name="Sample - Superstore.csv", save_csv=True):
    """
    Télécharge, nettoie et retourne le DataFrame prêt à l'emploi.
    Si save_csv=True, sauvegarde également dans data/clean_superstore.csv
    """
    df_raw = load_dataset(dataset_name, file_name)
    df_cleaned = pipeline(df_raw)
    
    if save_csv:
        # Création du dossier data si inexistant
        if not os.path.exists("../data"):
            os.makedirs("../data")
        df_cleaned.to_csv("../data/clean_superstore.csv", index=False)
        print("Data saved to ../data/clean_superstore.csv")
    
    return df_cleaned
