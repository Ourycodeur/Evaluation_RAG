TEST_SET = [

    # =========================
    # QUESTIONS SIMPLES
    # =========================

    {
        "category": "simple",
        "question": "Quel joueur a marqué le plus de points cette saison ?",
        "ground_truth": "Le joueur ayant marqué le plus de points est SGA(Shaï Gilgeous- Alexander)"
    },

    {
        "category": "simple",
        "question": "Quel joueur a le plus de rebonds ?",
        "ground_truth": "Le joueur ayant le plus de rebonds est Nikola Jokic"
    },

    {
        "category": "simple",
        "question": "Quel joueur a le plus de passes décisives ?",
        "ground_truth": "Le joueur ayant le plus de passes décisives est Nikola Jokic"
    },

    # =========================
    # QUESTIONS COMPLEXES
    # =========================

    {
        "category": "complex",
        "question": "Quel joueur possède le meilleur pourcentage à trois points sur les cinq derniers matchs ?",
        "ground_truth": "D'après le contexte j'ai pas accès aux statistiques spécifiques des cinq derniers match"
    },

    {
        "category": "complex",
        "question": "Quel joueur combine le plus de points, rebonds et passes décisives ?",
        "ground_truth": "D'après le contexte fournis le joueur le plus polyvalent est Nikola Jokic"
    },

    {
        "category": "complex",
        "question": "Quelle équipe possède la meilleure efficacité offensive ?",
        "ground_truth": "D'après le contexte fournis l'equipe qui a la meilleure capacité offensive est Denver Nuggets"
    },

    # =========================
    # QUESTIONS BRUITÉES
    # =========================

    {
        "category": "noisy",
        "question": "c ki le meilleur shooteur a 3 pts ?",
        "ground_truth": "Le meilleur shooter à 3 points est Tyrese Haliburton"
    }
]