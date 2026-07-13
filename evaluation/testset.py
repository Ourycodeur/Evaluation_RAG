# evaluation/testset_com.py
"""
Testset PARTAGÉ entre l'évaluation Étape 1 (baseline, RAG seul) et l'évaluation
Étape 3 (RAG + Tool SQL). Un seul et même TEST_SET importé par les deux scripts,
pour garantir une comparaison avant/après valide (mêmes questions, mêmes ground
truths — toute divergence fausserait le comparatif).

Ground truths vérifiées contre la base réelle (SQLite/Postgres) avant intégration
(cf. échanges précédents : 4 des 7 ground truths initiales étaient factuellement
fausses — Jokić crédité à tort de rebonds/passes, Denver à tort en tête de
l'efficacité offensive, Haliburton à tort en tête du 3P%).
"""

TEST_SET = [
    # --- QUESTIONS SIMPLES (statistiques factuelles) ---
    {
        "category": "simple",
        "question": "Quel joueur a marqué le plus de points cette saison ?",
        "ground_truth": "Le joueur ayant marqué le plus de points cette saison est Shai Gilgeous-Alexander (SGA), avec 2485 points.",
    },
    {
        "category": "simple",
        "question": "Quel joueur a le plus de rebonds et de passes décisives ?",
        "ground_truth": "Le joueur ayant le plus de rebonds cette saison est Nikola Jokic.",
    },
    {
        "category": "simple",
        "question": "Quel joueur a le plus de passes décisives ?",
        "ground_truth": "Le joueur ayant le plus de passes décisives cette saison est Nikola Jokic,.",
    },
]