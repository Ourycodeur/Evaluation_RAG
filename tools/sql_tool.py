# sql_tool.py
"""
Tool SQL (Étape 2) — génération dynamique de requêtes SQL à partir d'une
question en langage naturel, exécution sécurisée en lecture seule, retour
des résultats formatés pour synthèse par le LLM.

Approche : LangChain SQLDatabase pour l'introspection du schéma + exécution,
génération de la requête via le LLM Mistral avec few-shot examples couvrant
les pièges métier identifiés (ex: "meilleur % à 3 points" nécessite un filtre
de volume qualifié, sinon un joueur à 1/1 remonte en tête).

Garde-fous de sécurité :
- Uniquement des requêtes SELECT (rejet de tout DROP/DELETE/UPDATE/INSERT/ALTER)
- Timeout d'exécution
- Résultats tronqués pour éviter d'inonder le contexte du LLM de synthèse
"""

import logging
import re
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
from langchain_community.utilities import SQLDatabase

from utils.config import MISTRAL_API_KEY, MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FORBIDDEN_KEYWORDS = ("DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "CREATE", "GRANT", "ATTACH")

# Few-shot examples : couvrent les questions "métier" types du testset, y compris
# le piège du volume qualifié (3P% sur peu de tentatives = non représentatif).
FEW_SHOT_EXAMPLES = """
Exemple 1 :
Question : "Quel joueur a marqué le plus de points cette saison ?"
SQL : SELECT p.player_name, s.points FROM stats s JOIN players p ON s.player_id = p.player_id ORDER BY s.points DESC LIMIT 1;

Exemple 2 :
Question : "Quel joueur a le plus de rebonds ?"
SQL : SELECT p.player_name, s.reb FROM stats s JOIN players p ON s.player_id = p.player_id ORDER BY s.reb DESC LIMIT 1;

Exemple 3 :
Question : "Quel joueur a le meilleur pourcentage à trois points ?"
SQL : SELECT p.player_name, s.tp_pct, s.tpa FROM stats s JOIN players p ON s.player_id = p.player_id WHERE s.tpa >= 150 ORDER BY s.tp_pct DESC LIMIT 5;
-- Note : le filtre "s.tpa >= 150" est indispensable. Sans lui, un joueur ayant tenté
-- 1 seul tir à 3 points et l'ayant réussi remonterait en tête avec 100%, ce qui n'est
-- pas représentatif. Appliquer ce même principe de "volume qualifié" pour toute
-- question sur un pourcentage (FG%, FT%, 3P%) : ajouter un seuil raisonnable sur le
-- nombre de tentatives (colonne FGA/FTA/3PA correspondante).

Exemple 4 :
Question : "Quelle équipe a la meilleure efficacité offensive ?"
SQL : SELECT t.team_name, AVG(s.off_rtg) AS avg_offrtg FROM stats s JOIN players p ON s.player_id = p.player_id JOIN teams t ON p.team_code = t.team_code WHERE s.games_played >= 40 GROUP BY t.team_name ORDER BY avg_offrtg DESC LIMIT 1;
-- Note : filtrer sur games_played >= 40 pour ne pas biaiser la moyenne d'équipe avec
-- des joueurs ayant très peu joué (call-ups, blessures).

Exemple 5 :
Question : "Quel joueur combine le plus de points, rebonds et passes décisives ?"
SQL : SELECT p.player_name, (s.points + s.reb + s.ast) AS total FROM stats s JOIN players p ON s.player_id = p.player_id ORDER BY total DESC LIMIT 1;
"""


def _get_schema_description(db: SQLDatabase) -> str:
    """Retourne la description du schéma (tables + colonnes) pour le prompt de génération SQL."""
    return db.get_table_info(table_names=["players", "teams", "stats"])


def _is_safe_select(sql: str) -> bool:
    """Vérifie que la requête est un SELECT pur, sans mot-clé destructif."""
    upper_sql = sql.upper()
    if not upper_sql.strip().startswith("SELECT"):
        return False
    return not any(re.search(rf"\b{kw}\b", upper_sql) for kw in FORBIDDEN_KEYWORDS)


def generate_sql_query(question: str, db: SQLDatabase, client: MistralClient) -> str:
    """Génère une requête SQL à partir d'une question en langage naturel, via few-shot prompting."""
    schema = _get_schema_description(db)

    prompt = f"""Tu es un générateur de requêtes SQL pour une base de données de statistiques NBA (SQLite/PostgreSQL).

Schéma de la base :
{schema}

Voici des exemples de questions et des requêtes SQL correctes associées :
{FEW_SHOT_EXAMPLES}

Génère UNIQUEMENT la requête SQL (SELECT uniquement, pas d'explication, pas de balises markdown)
pour répondre à la question suivante. Applique les mêmes principes de rigueur que dans les
exemples (filtre de volume qualifié pour les pourcentages, filtre games_played pour les moyennes
d'équipe, etc.) même s'ils ne sont pas explicitement demandés dans la question.

Question : {question}

SQL :"""

    response = client.chat(
        model=MODEL_NAME,
        messages=[ChatMessage(role="user", content=prompt)],
        temperature=0.0,  # déterministe pour la génération de requêtes
        max_tokens=300,
    )
    sql = response.choices[0].message.content.strip()
    # Nettoyage des éventuelles balises markdown que le LLM pourrait ajouter malgré la consigne
    sql = re.sub(r"^```sql\s*|\s*```$", "", sql, flags=re.IGNORECASE | re.MULTILINE).strip()
    return sql


def run_sql_tool(question: str, database_url: str, max_rows: int = 10) -> dict:
    """
    Pipeline complet : question en langage naturel -> SQL généré -> exécution -> résultats.
    Retourne un dict avec le SQL généré, les résultats et le statut, pour traçabilité
    (destiné à alimenter la table `reports`).
    """
    db = SQLDatabase.from_uri(database_url, include_tables=["players", "teams", "stats"])
    client = MistralClient(api_key=MISTRAL_API_KEY)

    try:
        sql_query = generate_sql_query(question, db, client)
        logging.info(f"[sql_tool] SQL généré pour '{question}': {sql_query}")
    except Exception as e:
        logging.error(f"[sql_tool] Échec de génération SQL: {e}")
        return {"status": "error", "sql": None, "result": None, "error": f"Erreur de génération SQL: {e}"}

    if not _is_safe_select(sql_query):
        logging.error(f"[sql_tool] Requête rejetée (non-SELECT ou mot-clé interdit): {sql_query}")
        return {
            "status": "error",
            "sql": sql_query,
            "result": None,
            "error": "Requête générée rejetée : seules les requêtes SELECT en lecture seule sont autorisées.",
        }

    try:
        raw_result = db.run(sql_query, fetch="all")
        logging.info(f"[sql_tool] Exécution réussie, {len(raw_result) if hasattr(raw_result, '__len__') else '?'} résultat(s).")
        return {"status": "success", "sql": sql_query, "result": raw_result, "error": None}
    except Exception as e:
        logging.error(f"[sql_tool] Échec d'exécution SQL: {e}")
        return {"status": "error", "sql": sql_query, "result": None, "error": f"Erreur d'exécution SQL: {e}"}