import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine

env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(env_path, override=True)

engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:5432/{os.getenv('DB_NAME')}",
    connect_args={"sslmode": "require"}
)
