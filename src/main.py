import sys
from pathlib import Path

sys.path.append(str(Path().resolve().parents[1]))


from data_loader import get_clean_data
from src.db import engine

# 1️⃣ Charger et nettoyer les données
df_clean = get_clean_data()

# 2️⃣ Connexion à RDS
engine = get_engine()

# 3️⃣ Envoyer les données vers la base RDS
df_clean.to_sql("cleaned_data", engine, if_exists="replace", index=False)
print("Data uploaded to RDS successfully!")

# 4️⃣  Vérification   
import pandas as pd
df_check = pd.read_sql("SELECT * FROM cleaned_data LIMIT 5", engine)
print(df_check)