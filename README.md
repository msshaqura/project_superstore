# Superstore ETL Project

## Vue d'ensemble du projet
Ce projet implémente un pipeline ETL pour charger les données Superstore depuis un CSV, les nettoyer, et créer un schéma en étoile (star schema) dans PostgreSQL en utilisant SQLAlchemy et Pandas.

## Prérequis
- Python 3.11+
- PostgreSQL 15+
- Docker & Docker Compose
- Librairies Python :
  - pandas
  - sqlalchemy
  - psycopg2
  - python-dotenv
  - chardet
  - kagglehub

## Instructions d'installation

### 1️⃣ Créer et activer un environnement virtuel
```bash
python -m venv venv
# Linux / Mac
source venv/bin/activate
# Windows
venv\Scripts\activate


## Installer les dépendances
pip install -r requirements.txt

## Démarrer PostgreSQL avec Docker Compose
docker-compose up -d

## Lancer le script ETL principal
python src/main.py


## Structure du projet
Bloc_6/
│
├─ data/
│  └─ clean_superstore.csv  # CSV nettoyé prêt pour la DB
│
├─ database/
│  ├─ schema.sql          
│  ├─ schema.sql.save  
│
├─ notebooks/
│  ├─ check_db.ipynb
│  ├─ sample_superstore.ipynb  
│
├─ src/
│  ├─ main.py          # Script principal pour exécuter l'ETL
│  ├─ db.py            # Configuration de la base de données et création du moteur SQLAlchemy
│  ├─ data_loader.py   # Téléchargement et nettoyage des données
│  └─ load_to_db.py    # Chargement des données dans les tables PostgreSQL
│
├─ docker-compose.yml
└─ README.md


# Pour démarrer docker : docker compose up
# Pour démarrer RDS : ENV_FILE=.env docker compose up -d