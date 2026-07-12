CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_name TEXT NOT NULL,
    team TEXT,
    age INTEGER
);

CREATE TABLE matches (
    match_id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_played REAL,
    game_losed REAL
);

CREATE TABLE stats (
    stat_id INTEGER PRIMARY KEY AUTOINCREMENT,

    player_id INTEGER,
    match_id INTEGER,

    pts REAL,
    reb REAL,
    ast REAL,

    fg_pct REAL,
    fg3_pct REAL,
    ft_pct REAL,

    FOREIGN KEY(player_id)
        REFERENCES players(player_id),

    FOREIGN KEY(match_id)
        REFERENCES matches(match_id)
);

CREATE TABLE reports (
    report_id INTEGER PRIMARY KEY AUTOINCREMENT,

    title TEXT,
    source TEXT,
    content TEXT
);