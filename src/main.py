from db import wait_for_db, engine
from data_loader import get_clean_data
from load_to_db import load_dim_tables, load_fact_tables
import pandas as pd

print("💻 Initialisation...")

# 1️⃣ Attendre DB
wait_for_db()

# 2️⃣ Charger et nettoyer les données
df_clean = get_clean_data()

# 3️⃣ Créer dim_date
dates = pd.date_range(start=df_clean['order_date'].min(), end=df_clean['order_date'].max())
df_dim_date = pd.DataFrame({
    'date_id': dates,
    'year': dates.year,
    'quarter': dates.quarter,
    'month': dates.month,
    'month_name': dates.strftime('%B'),
    'week': dates.isocalendar().week,
    'day': dates.day,
    'day_name': dates.strftime('%A'),
    'is_weekend': dates.weekday >= 5
})

# 4️⃣ Charger les tables
load_dim_tables(df_clean, df_dim_date)
load_fact_tables(df_clean)

print("✅ Tout est prêt !")
