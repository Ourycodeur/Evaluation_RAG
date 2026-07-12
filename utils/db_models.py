# db/models_db.py
"""
Modélisation relationnelle de la base NBA Analyst AI (Étape 2).

Tables demandées dans le brief : players, matches, stats, reports.
Table ajoutée par nécessité technique : teams (référencée en clé étrangère par
players/matches, indispensable pour l'intégrité relationnelle — le brief ne
mentionne pas cette table explicitement mais players.team et matches.home/away
ne peuvent pas exister sans elle).

Compatible PostgreSQL ET SQLite via la même définition SQLAlchemy (le moteur
cible est déterminé par DATABASE_URL, cf. utils/config.py). SQLite est utilisé
ici pour la validation locale du pipeline ; PostgreSQL est la cible recommandée
pour la mise en production (cf. échange précédent sur la robustesse aux accès
concurrents du Tool SQL).
"""

from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, SmallInteger, Numeric, Date, DateTime,
    ForeignKey, Text, UniqueConstraint, CheckConstraint, Index
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Team(Base):
    """Table de référence des équipes NBA (ajoutée pour l'intégrité des FK)."""
    __tablename__ = "teams"

    team_code = Column(String(3), primary_key=True)   # ex: "OKC"
    team_name = Column(String(50), nullable=False)     # ex: "Oklahoma City Thunder"

    players = relationship("Player", back_populates="team")


class Player(Base):
    """Table players : identité et rattachement équipe/saison d'un joueur."""
    __tablename__ = "players"

    player_id = Column(Integer, primary_key=True, autoincrement=True)
    player_name = Column(String(100), nullable=False)
    team_code = Column(String(3), ForeignKey("teams.team_code"), nullable=True)
    age = Column(SmallInteger, nullable=True)
    season = Column(String(9), nullable=False, default="2025-2026")  # ex: "2025-2026"

    team = relationship("Team", back_populates="players")
    stats = relationship("Stat", back_populates="player")

    __table_args__ = (
        # Un même joueur ne doit apparaître qu'une fois par saison
        UniqueConstraint("player_name", "season", name="uq_player_season"),
        CheckConstraint("age IS NULL OR age > 0", name="ck_player_age_positive"),
    )


class Match(Base):
    """
    Table matches : rencontres NBA (date, équipes, score).

    ⚠️ NON ALIMENTÉE PAR LES DONNÉES ACTUELLES : le fichier regular_NBA.xlsx ne
    contient que des agrégats saison, aucune donnée par match. Cette table est
    créée avec un schéma complet pour respecter le brief et anticiper une future
    source de données (ex: export calendrier/box-scores officiel NBA).
    """
    __tablename__ = "matches"

    match_id = Column(Integer, primary_key=True, autoincrement=True)
    season = Column(String(9), nullable=False)
    match_date = Column(Date, nullable=True)
    home_team_code = Column(String(3), ForeignKey("teams.team_code"), nullable=True)
    away_team_code = Column(String(3), ForeignKey("teams.team_code"), nullable=True)
    home_score = Column(SmallInteger, nullable=True)
    away_score = Column(SmallInteger, nullable=True)

    stats = relationship("Stat", back_populates="match")

    __table_args__ = (
        CheckConstraint("home_team_code != away_team_code", name="ck_match_distinct_teams"),
    )


class Stat(Base):
    """
    Table stats : statistiques de performance d'un joueur.

    match_id est NULLABLE : avec les données actuelles (agrégats saison), chaque
    ligne représente une saison complète pour un joueur (match_id = NULL). Le
    champ est prévu pour accueillir de futures stats par match sans changer le
    schéma, une fois une source de données adaptée disponible.
    """
    __tablename__ = "stats"

    stat_id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, ForeignKey("players.player_id"), nullable=False)
    match_id = Column(Integer, ForeignKey("matches.match_id"), nullable=True)
    season = Column(String(9), nullable=False)

    games_played = Column(SmallInteger)
    wins = Column(SmallInteger)
    losses = Column(SmallInteger)
    minutes = Column(Numeric(6, 1))
    points = Column(Numeric(7, 1))
    fgm = Column(Numeric(6, 1))
    fga = Column(Numeric(6, 1))
    fg_pct = Column(Numeric(5, 2))
    tpm = Column(Numeric(6, 1))          # colonne corrigée (ex "15:00:00")
    tpa = Column(Numeric(6, 1))
    tp_pct = Column(Numeric(5, 2))
    ftm = Column(Numeric(6, 1))
    fta = Column(Numeric(6, 1))
    ft_pct = Column(Numeric(5, 2))
    oreb = Column(Numeric(6, 1))
    dreb = Column(Numeric(6, 1))
    reb = Column(Numeric(6, 1))
    ast = Column(Numeric(6, 1))
    tov = Column(Numeric(6, 1))
    stl = Column(Numeric(6, 1))
    blk = Column(Numeric(6, 1))
    pf = Column(Numeric(6, 1))
    fp = Column(Numeric(7, 1))
    dd2 = Column(SmallInteger)
    td3 = Column(SmallInteger)
    plus_minus = Column(Numeric(6, 1))
    off_rtg = Column(Numeric(6, 1))
    def_rtg = Column(Numeric(6, 1))
    net_rtg = Column(Numeric(6, 1))
    ast_pct = Column(Numeric(5, 2))
    ast_to = Column(Numeric(5, 2))
    ast_ratio = Column(Numeric(5, 2))
    oreb_pct = Column(Numeric(5, 2))
    dreb_pct = Column(Numeric(5, 2))
    reb_pct = Column(Numeric(5, 2))
    to_ratio = Column(Numeric(5, 2))
    efg_pct = Column(Numeric(5, 2))
    ts_pct = Column(Numeric(5, 2))
    usg_pct = Column(Numeric(5, 2))
    pace = Column(Numeric(6, 1))
    pie = Column(Numeric(5, 2))
    poss = Column(Numeric(7, 1))

    player = relationship("Player", back_populates="stats")
    match = relationship("Match", back_populates="stats")

    __table_args__ = (
        # Empêche la double insertion d'un agrégat saison pour le même joueur
        # (contrainte partielle : ne s'applique qu'aux lignes match_id IS NULL)
        Index(
            "uq_stats_player_season_aggregate",
            "player_id", "season",
            unique=True,
            sqlite_where=Column("match_id").is_(None),
            postgresql_where=Column("match_id").is_(None),
        ),
        CheckConstraint("fgm IS NULL OR fga IS NULL OR fgm <= fga", name="ck_fgm_le_fga"),
        CheckConstraint("tpm IS NULL OR fgm IS NULL OR tpm <= fgm", name="ck_tpm_le_fgm"),
        CheckConstraint("ftm IS NULL OR fta IS NULL OR ftm <= fta", name="ck_ftm_le_fta"),
    )


class Report(Base):
    """
    Table reports : traçabilité des interactions du Tool SQL (Étape 2/3).

    Hypothèse posée en l'absence de définition dans le brief : chaque appel du
    Tool SQL par l'agent (question utilisateur -> SQL généré -> résultat ->
    réponse finale du LLM) est journalisé ici. Sert de base à la traçabilité
    demandée dans le cadrage général de la mission, et à l'analyse comparative
    de l'Étape 3.
    """
    __tablename__ = "reports"

    report_id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_question = Column(Text, nullable=False)
    generated_sql = Column(Text, nullable=True)
    sql_result_summary = Column(Text, nullable=True)
    final_answer = Column(Text, nullable=True)
    execution_status = Column(String(20), nullable=False, default="success")  # success | error
    error_message = Column(Text, nullable=True)
    latency_ms = Column(Integer, nullable=True)