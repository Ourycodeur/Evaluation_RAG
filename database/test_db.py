"""
import sqlite3

conn = sqlite3.connect("../database/nba.db")

cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM players")
print("Players :", cursor.fetchone()[0])

cursor.execute("SELECT * FROM players LIMIT 5")
print(cursor.fetchall())

conn.close()"""

import sqlite3

conn = sqlite3.connect("../database/nba.db")
cursor = conn.cursor() 

cursor.execute("""
SELECT player_id,
       MAX(points)
FROM stats
""")

print(cursor.fetchall())

print(cursor.fetchall())