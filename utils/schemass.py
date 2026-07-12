# utils/schemas.py
"""
Modèles Pydantic de validation des lignes issues de regular_NBA.xlsx avant
insertion en base (Étape 2 - sécurisation du flux d'ingestion).

Objectif : détecter et rejeter/signaler toute ligne incohérente AVANT qu'elle
n'atteigne la base de données (ex: pourcentage hors [0,100], FGM > FGA, valeurs
négatives), plutôt que de laisser SQL Server/Postgres découvrir le problème via
une contrainte CHECK en échec silencieux ou pire, une insertion silencieusement
fausse.
"""

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class ExcelPlayerStatRow(BaseModel):
    """Validation d'une ligne brute de la feuille 'Données NBA'."""

    player_name: str = Field(..., min_length=1)
    team_code: Optional[str] = Field(None, min_length=2, max_length=3)
    age: Optional[int] = Field(None, gt=0, lt=50)
    games_played: Optional[int] = Field(None, ge=0, le=82)
    wins: Optional[int] = Field(None, ge=0)
    losses: Optional[int] = Field(None, ge=0)
    minutes: Optional[float] = Field(None, ge=0)
    points: Optional[float] = Field(None, ge=0)
    fgm: Optional[float] = Field(None, ge=0)
    fga: Optional[float] = Field(None, ge=0)
    fg_pct: Optional[float] = Field(None, ge=0, le=100)
    tpm: Optional[float] = Field(None, ge=0)
    tpa: Optional[float] = Field(None, ge=0)
    tp_pct: Optional[float] = Field(None, ge=0, le=100)
    ftm: Optional[float] = Field(None, ge=0)
    fta: Optional[float] = Field(None, ge=0)
    ft_pct: Optional[float] = Field(None, ge=0, le=100)
    oreb: Optional[float] = Field(None, ge=0)
    dreb: Optional[float] = Field(None, ge=0)
    reb: Optional[float] = Field(None, ge=0)
    ast: Optional[float] = Field(None, ge=0)
    tov: Optional[float] = Field(None, ge=0)
    stl: Optional[float] = Field(None, ge=0)
    blk: Optional[float] = Field(None, ge=0)
    pf: Optional[float] = Field(None, ge=0)
    fp: Optional[float] = Field(None)
    dd2: Optional[int] = Field(None, ge=0)
    td3: Optional[int] = Field(None, ge=0)
    plus_minus: Optional[float] = Field(None)
    off_rtg: Optional[float] = Field(None)
    def_rtg: Optional[float] = Field(None)
    net_rtg: Optional[float] = Field(None)
    ast_pct: Optional[float] = Field(None)
    ast_to: Optional[float] = Field(None)
    ast_ratio: Optional[float] = Field(None)
    oreb_pct: Optional[float] = Field(None, ge=0, le=100)
    dreb_pct: Optional[float] = Field(None, ge=0, le=100)
    reb_pct: Optional[float] = Field(None, ge=0, le=100)
    to_ratio: Optional[float] = Field(None)
    efg_pct: Optional[float] = Field(None, ge=0, le=150)  # eFG% peut dépasser 100% par construction
    ts_pct: Optional[float] = Field(None, ge=0, le=150)   # idem TS%
    usg_pct: Optional[float] = Field(None, ge=0, le=100)
    pace: Optional[float] = Field(None, ge=0)
    pie: Optional[float] = Field(None)
    poss: Optional[float] = Field(None, ge=0)

    @field_validator("team_code")
    @classmethod
    def uppercase_team_code(cls, v):
        return v.upper().strip() if v else v

    @field_validator("player_name")
    @classmethod
    def strip_player_name(cls, v):
        return v.strip()

    @model_validator(mode="after")
    def check_fgm_le_fga(self):
        if self.fgm is not None and self.fga is not None and self.fgm > self.fga:
            raise ValueError(f"Incohérence: FGM ({self.fgm}) > FGA ({self.fga})")
        return self

    @model_validator(mode="after")
    def check_tpm_le_fgm(self):
        if self.tpm is not None and self.fgm is not None and self.tpm > self.fgm:
            raise ValueError(f"Incohérence: 3PM ({self.tpm}) > FGM ({self.fgm})")
        return self

    @model_validator(mode="after")
    def check_ftm_le_fta(self):
        if self.ftm is not None and self.fta is not None and self.ftm > self.fta:
            raise ValueError(f"Incohérence: FTM ({self.ftm}) > FTA ({self.fta})")
        return self

    @model_validator(mode="after")
    def check_wins_losses_le_games(self):
        if self.games_played is not None and self.wins is not None and self.losses is not None:
            if self.wins + self.losses > self.games_played:
                raise ValueError(
                    f"Incohérence: W+L ({self.wins + self.losses}) > GP ({self.games_played})"
                )
        return self


class TeamRow(BaseModel):
    """Validation d'une ligne de la feuille 'Equipe' (mapping code -> nom)."""

    team_code: str = Field(..., min_length=2, max_length=3)
    team_name: str = Field(..., min_length=1)

    @field_validator("team_code")
    @classmethod
    def uppercase_code(cls, v):
        return v.upper().strip()

    @field_validator("team_name")
    @classmethod
    def strip_name(cls, v):
        return v.strip()