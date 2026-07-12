# load_excel_to_db.py
"""
Pipeline d'ingestion Excel -> Base de données (Étape 2).

Étapes : lecture Excel -> validation Pydantic (schemas.py) -> insertion via
SQLAlchemy (models_db.py). Toute ligne invalide est journalisée et exclue de
l'insertion plutôt que de faire planter tout le batch (cohérent avec la
stratégie de gestion d'erreurs déjà en place sur le pipeline RAG).

Usage :
    python load_excel_to_db.py --excel-path regular_NBA.xlsx --season 2025-2026

DATABASE_URL par défaut : SQLite local (db/nba_analyst.db) pour la validation
rapide du pipeline. Pointer vers une URL PostgreSQL (ex:
"postgresql://user:pass@host:5432/nba_db") en production, sans changer le code
(SQLAlchemy est agnostique du moteur).
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.db_models import Base, Team, Player, Stat
from utils.schemass import ExcelPlayerStatRow, TeamRow

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DATABASE_URL = "sqlite:///database/nba.db"

# Mapping colonnes Excel (nom brut, incluant la colonne corrompue) -> champs du modèle Pydantic
COLUMN_MAPPING = {
    "Player": "player_name",
    "Team": "team_code",
    "Age": "age",
    "GP": "games_played",
    "W": "wins",
    "L": "losses",
    "Min": "minutes",
    "PTS": "points",
    "FGM": "fgm",
    "FGA": "fga",
    "FG%": "fg_pct",
    # La colonne d'origine "3PM" a été auto-convertie en heure par Excel (cf. diagnostic
    # précédent) ; son nom réel dans le DataFrame varie selon la version de pandas/openpyxl
    # (objet datetime.time(15, 0) ou chaîne "15:00:00"). On la retrouve par position (index 11)
    # plutôt que par nom, cf. fix_corrupted_3pm_header().
    "3PA": "tpa",
    "3P%": "tp_pct",
    "FTM": "ftm",
    "FTA": "fta",
    "FT%": "ft_pct",
    "OREB": "oreb",
    "DREB": "dreb",
    "REB": "reb",
    "AST": "ast",
    "TOV": "tov",
    "STL": "stl",
    "BLK": "blk",
    "PF": "pf",
    "FP": "fp",
    "DD2": "dd2",
    "TD3": "td3",
    "+/-": "plus_minus",
    "OFFRTG": "off_rtg",
    "DEFRTG": "def_rtg",
    "NETRTG": "net_rtg",
    "AST%": "ast_pct",
    "AST/TO": "ast_to",
    "AST RATIO": "ast_ratio",
    "OREB%": "oreb_pct",
    "DREB%": "dreb_pct",
    "REB%": "reb_pct",
    "TO RATIO": "to_ratio",
    "EFG%": "efg_pct",
    "TS%": "ts_pct",
    "USG%": "usg_pct",
    "PACE": "pace",
    "PIE": "pie",
    "POSS": "poss",
}


def fix_corrupted_3pm_header(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomme par position la colonne "3PM" corrompue par l'autoformatage Excel
    (cf. diagnostic : la colonne 11 contient les vraies valeurs de 3PM, seul son
    nom a été altéré en objet heure "15:00:00"). Les valeurs sous-jacentes ne
    sont PAS modifiées, seul le nom de colonne est corrigé.
    """
    col_index = 11
    original_name = df.columns[col_index]
    df = df.rename(columns={original_name: "3PM"})
    logging.info(f"Colonne à l'index {col_index} renommée de '{original_name}' vers '3PM' (correction du bug Excel).")
    return df


def load_and_validate_players(excel_path: str, season: str) -> tuple[list[ExcelPlayerStatRow], list[dict]]:
    """Lit la feuille 'Données NBA', corrige l'en-tête 3PM, et valide chaque ligne via Pydantic."""
    df = pd.read_excel(excel_path, sheet_name="Données NBA", header=0)
    df.columns = df.iloc[0]
    df = df[1:].reset_index(drop=True)
    df = fix_corrupted_3pm_header(df)

    full_mapping = {**COLUMN_MAPPING, "3PM": "tpm"}

    valid_rows: list[ExcelPlayerStatRow] = []
    errors: list[dict] = []

    for i, row in df.iterrows():
        raw = {}
        for excel_col, model_field in full_mapping.items():
            if excel_col not in df.columns:
                continue
            value = row[excel_col]
            raw[model_field] = None if pd.isna(value) else value

        try:
            validated = ExcelPlayerStatRow(**raw)
            valid_rows.append(validated)
        except ValidationError as e:
            errors.append({
                "row_index": i,
                "player_name": raw.get("player_name", "INCONNU"),
                "errors": e.errors(),
            })

    logging.info(f"Validation terminée : {len(valid_rows)} lignes valides, {len(errors)} lignes rejetées.")
    if errors:
        for err in errors[:10]:  # on n'affiche que les 10 premières pour ne pas noyer les logs
            logging.warning(f"  Ligne rejetée ({err['player_name']}): {err['errors']}")
        if len(errors) > 10:
            logging.warning(f"  ... et {len(errors) - 10} autres lignes rejetées (voir rapport complet).")

    return valid_rows, errors


def load_and_validate_teams(excel_path: str) -> tuple[list[TeamRow], list[dict]]:
    """Lit la feuille 'Equipe' et valide chaque ligne via Pydantic."""
    df = pd.read_excel(excel_path, sheet_name="Equipe")
    valid_rows: list[TeamRow] = []
    errors: list[dict] = []

    for i, row in df.iterrows():
        try:
            validated = TeamRow(team_code=row["Code"], team_name=row["Nom complet de l'équipe"])
            valid_rows.append(validated)
        except ValidationError as e:
            errors.append({"row_index": i, "errors": e.errors()})

    logging.info(f"Équipes : {len(valid_rows)} valides, {len(errors)} rejetées.")
    return valid_rows, errors


def insert_into_db(session, teams: list[TeamRow], players: list[ExcelPlayerStatRow], season: str) -> dict:
    """Insère les équipes, joueurs et stats validés dans la base, en gérant les doublons."""
    summary = {"teams_inserted": 0, "players_inserted": 0, "stats_inserted": 0, "skipped_duplicates": 0}

    # 1. Équipes
    for team in teams:
        exists = session.query(Team).filter_by(team_code=team.team_code).first()
        if not exists:
            session.add(Team(team_code=team.team_code, team_name=team.team_name))
            summary["teams_inserted"] += 1
    session.commit()

    # 2. Joueurs + stats (une transaction par joueur pour isoler les échecs)
    for p in players:
        existing_player = session.query(Player).filter_by(player_name=p.player_name, season=season).first()
        if existing_player:
            summary["skipped_duplicates"] += 1
            continue

        try:
            player_obj = Player(
                player_name=p.player_name,
                team_code=p.team_code,
                age=p.age,
                season=season,
            )
            session.add(player_obj)
            session.flush()  # récupère player_id sans commit complet

            stat_obj = Stat(
                player_id=player_obj.player_id,
                match_id=None,  # agrégat saison, pas de match associé (cf. limite documentée)
                season=season,
                games_played=p.games_played, wins=p.wins, losses=p.losses,
                minutes=p.minutes, points=p.points,
                fgm=p.fgm, fga=p.fga, fg_pct=p.fg_pct,
                tpm=p.tpm, tpa=p.tpa, tp_pct=p.tp_pct,
                ftm=p.ftm, fta=p.fta, ft_pct=p.ft_pct,
                oreb=p.oreb, dreb=p.dreb, reb=p.reb,
                ast=p.ast, tov=p.tov, stl=p.stl, blk=p.blk, pf=p.pf,
                fp=p.fp, dd2=p.dd2, td3=p.td3, plus_minus=p.plus_minus,
                off_rtg=p.off_rtg, def_rtg=p.def_rtg, net_rtg=p.net_rtg,
                ast_pct=p.ast_pct, ast_to=p.ast_to, ast_ratio=p.ast_ratio,
                oreb_pct=p.oreb_pct, dreb_pct=p.dreb_pct, reb_pct=p.reb_pct,
                to_ratio=p.to_ratio, efg_pct=p.efg_pct, ts_pct=p.ts_pct,
                usg_pct=p.usg_pct, pace=p.pace, pie=p.pie, poss=p.poss,
            )
            session.add(stat_obj)
            session.commit()
            summary["players_inserted"] += 1
            summary["stats_inserted"] += 1
        except IntegrityError as e:
            session.rollback()
            logging.error(f"Échec insertion joueur '{p.player_name}': {e}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Ingestion Excel -> DB pour NBA Analyst AI")
    parser.add_argument("--excel-path", required=True, help="Chemin vers regular_NBA.xlsx")
    parser.add_argument("--season", default="2025-2026", help="Saison à associer aux données (ex: 2025-2026)")
    parser.add_argument("--database-url", default=os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL))
    args = parser.parse_args()

    os.makedirs("db", exist_ok=True)
    engine = create_engine(args.database_url)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    logging.info(f"=== Ingestion démarrée : {args.excel_path} (saison {args.season}) ===")

    teams, team_errors = load_and_validate_teams(args.excel_path)
    players, player_errors = load_and_validate_players(args.excel_path, args.season)

    summary = insert_into_db(session, teams, players, args.season)

    total_errors = len(team_errors) + len(player_errors)
    logging.info("=== Résumé de l'ingestion ===")
    logging.info(f"Équipes insérées      : {summary['teams_inserted']}")
    logging.info(f"Joueurs insérés       : {summary['players_inserted']}")
    logging.info(f"Lignes stats insérées : {summary['stats_inserted']}")
    logging.info(f"Doublons ignorés      : {summary['skipped_duplicates']}")
    logging.info(f"Lignes rejetées (validation Pydantic) : {total_errors}")

    session.close()
    return summary, team_errors + player_errors


if __name__ == "__main__":
    main()