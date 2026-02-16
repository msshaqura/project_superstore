import os
import time
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

# ---------------------------------------
# 1️⃣ Charger les variables d'environnement
# ---------------------------------------
env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path)

DB_HOST = os.getenv("DB_HOST")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")
ENV = os.getenv("ENV", "dev")

# ---------------------------------------
# 2️⃣ Vérification de sécurité
# ---------------------------------------
if "amazonaws" in str(DB_HOST) and ENV != "prod":
    raise Exception(
        "❌ ACCÈS RDS BLOQUÉ ! Utilisez ENV=prod pour RDS."
    )

# ---------------------------------------
# 3️⃣ Création du moteur SQLAlchemy
# ---------------------------------------
def get_engine():
    db_url = f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"
    if "amazonaws" in str(DB_HOST):
        engine = create_engine(db_url, connect_args={"sslmode": "require"})
    else:
        engine = create_engine(db_url)
    return engine

engine = get_engine()

# ---------------------------------------
# 4️⃣ Fonction pour attendre DB
# ---------------------------------------
def wait_for_db(retries=10, delay=3):
    for i in range(retries):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ Database connection successful")
            return True
        except OperationalError:
            print(f"⏳ Waiting for database... ({i+1}/{retries})")
            time.sleep(delay)
    raise Exception("❌ Database is not responding!")
