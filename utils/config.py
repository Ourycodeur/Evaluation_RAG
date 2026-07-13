# utils/config.py
import os
from pathlib import Path
from dotenv import load_dotenv

# --- Racine du projet (indépendante du répertoire d'exécution / cwd) ---
BASE_DIR = Path(__file__).resolve().parent.parent

# Charger les variables d'environnement du fichier .env à la racine du projet
load_dotenv(BASE_DIR / ".env")

# --- Clé API ---
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    print("⚠️ Attention: La clé API Mistral (MISTRAL_API_KEY) n'est pas définie dans le fichier .env")

# --- Modèles Mistral ---
EMBEDDING_MODEL = "mistral-embed"
MODEL_NAME = "mistral-small-latest"

# --- Configuration de l'Indexation (tout ancré sur BASE_DIR, plus de chemin relatif) ---
INPUT_DIR = str(BASE_DIR / "inputs")
VECTOR_DB_DIR = str(BASE_DIR / "vector_db")
FAISS_INDEX_FILE = str(BASE_DIR / "vector_db" / "faiss_index.idx")
DOCUMENT_CHUNKS_FILE = str(BASE_DIR / "vector_db" / "document_chunks.pkl")

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 32

# --- Configuration de la Recherche ---
SEARCH_K = 5

# --- Base de données des logs d'interactions ---
DATABASE_DIR = BASE_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)  # garantit que le dossier existe
DATABASE_FILE = str(DATABASE_DIR / "interactions.db")
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# --- Base des données NBA (joueurs/équipes/stats) ---
# Emplacement confirmé : P10_DSML/database/nba.db
STATS_DATABASE_FILE = str(DATABASE_DIR / "nba.db")
STATS_DATABASE_URL = os.getenv("STATS_DATABASE_URL", f"sqlite:///{STATS_DATABASE_FILE}")

# --- Configuration de l'Application ---
APP_TITLE = "NBA Analyst AI"
NAME = "NBA"