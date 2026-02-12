from dotenv import load_dotenv
import os
from google.cloud import bigquery

load_dotenv()

client = bigquery.Client()
table_id = f"{os.getenv('PROJECT_ID')}.{os.getenv('DATASET')}.orders"

print("Connected successfully 🎉")

