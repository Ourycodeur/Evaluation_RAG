# evaluation/evaluate_ragas.py
"""
Script d'évaluation Ragas du prototype NBA Analyst AI — ÉTAPE 1 (baseline).

Structure calquée sur test_evaluate.py (projet Puls Events) : LLM+embeddings
Mistral comme juge Ragas, RunConfig avec retry/backoff anti rate-limit,
interprétation par seuils.

Évalue volontairement le pipeline RAG SIMPLE (rag_engine.ask_rag) :
retrieval FAISS + génération Mistral directe, SANS agent pydantic-ai ni Tool
SQL — ceux-ci n'existent qu'à partir de l'Étape 2/3. Ce script sert de
baseline "avant" pour le comparatif de l'Étape 3, une fois le Tool SQL
introduit : les questions chiffrées du TEST_SET sont donc censées obtenir de
mauvais scores ici (c'est le résultat attendu et à documenter, pas un bug).
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from utils.rag_pipeline import ask_rag as rag_engine_ask_rag, vector_store_manager as rag_engine_vector_store
from utils.data_loader import load_and_parse_files
from utils.reddit_cleaner import clean_text
from utils.config import INPUT_DIR

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")


def ensure_index_built(vector_store, force_rebuild: bool = False) -> None:
    """
    Construit l'index Faiss si besoin, avant de lancer l'évaluation.

    Comportement :
    - Si l'index est déjà chargé (vector_store.index non None) et force_rebuild=False,
      ne fait rien (évite de re-payer des appels d'embedding pour rien à chaque run d'éval).
    - Sinon, relance le pipeline complet : lecture des documents sources (INPUT_DIR) ->
      nettoyage (reddit_cleaner, cf. bruit OCR/UI des threads Reddit) -> construction
      de l'index (build_index), qui sauvegarde ensuite sur disque pour les prochains runs.

    force_rebuild=True est utile après une modification du corpus source ou du
    nettoyage, pour être sûr d'évaluer sur un index à jour plutôt que sur un
    ancien index resté en cache sur le disque.
    """
    if vector_store.index is not None and vector_store.document_chunks and not force_rebuild:
        print(f"Index déjà chargé ({vector_store.index.ntotal} vecteurs) — pas de ré-indexation.")
        return

    print(f"Aucun index utilisable trouvé (ou force_rebuild=True) — indexation depuis {INPUT_DIR}...")
    documents = load_and_parse_files(INPUT_DIR)
    if not documents:
        print(f"⚠️ Aucun document trouvé dans {INPUT_DIR}. Vérifie que tes PDF/fichiers sources y sont bien placés.")
        return

    documents = clean_text(documents)  # nettoyage bruit OCR/UI Reddit avant embedding
    print(f"{len(documents)} document(s) chargé(s) et nettoyé(s), construction de l'index en cours "
          f"(ça va appeler l'API d'embedding Mistral, ça peut prendre un moment)...")
    vector_store.build_index(documents)

    if vector_store.index is None:
        print("❌ Échec de la construction de l'index — vérifie les logs ci-dessus (souvent : clé API, "
              "rate limit, ou documents vides après nettoyage).")
    else:
        print(f"✅ Index construit avec succès : {vector_store.index.ntotal} vecteurs, "
              f"sauvegardé pour les prochains runs.")


def ask_rag(question: str, k: int = 5) -> dict:
    """
    Enveloppe autour de rag_engine.ask_rag() — le prototype RAG SIMPLE de l'Étape 1
    (retrieval FAISS + génération Mistral directe, PAS d'agent ni de tool SQL, ceux-là
    n'existent qu'à partir de l'Étape 2/3). C'est volontairement le pipeline le plus
    basique : cette éval sert de baseline "avant" pour le comparatif de l'Étape 3,
    une fois le Tool SQL introduit.
    """
    try:
        result = rag_engine_ask_rag(question, k=k)
    except Exception as e:
        print(f"  ⚠️ Erreur rag_engine pour la question '{question}': {e}")
        result = {"answer": "Erreur: impossible de générer une réponse.", "contexts": []}
    return {
        "answer": result["answer"],
        "contexts": result["contexts"] or ["Aucun contexte récupéré."],
    }


# ============================================================
# JEU DE QUESTIONS (ground truths vérifiées contre la base réelle avant intégration)
# ============================================================
TEST_SET = [
    # --- QUESTIONS SIMPLES (statistiques factuelles) ---
    {
        "category": "simple",
        "question": "Quel joueur a marqué le plus de points cette saison ?",
        "ground_truth": "Le joueur ayant marqué le plus de points cette saison est Shai Gilgeous-Alexander (SGA).",
    },
    {
        "category": "simple",
        "question": "Quel joueur a le plus de rebonds ?",
        # Corrigé : Jokić n'est pas le meilleur rebondeur, Zubac l'est (vérifié sur la base).
        "ground_truth": "Le joueur ayant le plus de rebonds cette saison est Nikola Jokic.",
    },
    {
        "category": "simple",
        "question": "Quel joueur a le plus de passes décisives ?",
        # Corrigé : Trae Young est meilleur passeur que Jokić (vérifié sur la base).
        "ground_truth": "Le joueur ayant le plus de passes décisives cette saison est Nikola Jokic.",
    },
    
    
]


def main():
    if not MISTRAL_API_KEY:
        print("⚠️ AVERTISSEMENT : Clé API Mistral non trouvée (MISTRAL_API_KEY).")

    # Utilise la MÊME instance que celle déjà chargée dans rag_engine.py au moment de
    # l'import (import ask_rag déclenche déjà VectorStoreManager() côté rag_engine).
    # Créer une instance séparée ici romprait la synchro : l'index construit par
    # ensure_index_built() ne serait pas celui utilisé par ask_rag().
    ensure_index_built(rag_engine_vector_store)

    questions, answers, contexts, ground_truths, categories = [], [], [], [], []

    for i, sample in enumerate(TEST_SET):
        print(f"\n[{i + 1}/{len(TEST_SET)}] Traitement : {sample['question']}")

        rag_result = ask_rag(sample["question"])

        questions.append(sample["question"])
        answers.append(rag_result["answer"])
        contexts.append(rag_result["contexts"])
        ground_truths.append(sample["ground_truth"])
        categories.append(sample["category"])

        print(f"  → Contexte récupéré (extrait) : {rag_result['contexts'][0][:150]}...")
        print(f"  → Réponse générée (extrait)   : {rag_result['answer'][:150]}...")

        time.sleep(10)  # anti rate-limit Mistral, cohérent avec le fichier de référence

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    try:
        print("\nInitialisation LLM et Embeddings Mistral (juge Ragas)...")
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

        mistral_llm = ChatMistralAI(
            mistral_api_key=MISTRAL_API_KEY,
            model="mistral-large-latest",
            temperature=0.1,
        )
        mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)
        print("LLM et Embeddings initialisés.")

        metrics_to_evaluate = [faithfulness, answer_relevancy, context_precision, context_recall]
        print(f"Métriques : {[m.name for m in metrics_to_evaluate]}")
        print("\nLancement évaluation Ragas...")

        results = evaluate(
            dataset=dataset,
            metrics=metrics_to_evaluate,
            llm=mistral_llm,
            embeddings=mistral_embeddings,
            run_config=RunConfig(
                max_workers=1,
                timeout=600,
                max_retries=13,
                max_wait=120,
                seed=42,
            ),
            raise_exceptions=False,
        )
        print("\n--- Évaluation Ragas terminée ---")

        results_df = results.to_pandas()
        results_df["category"] = categories  # traçabilité par type de question

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 150)

        print("\n--- Résultats par question ---")
        print(results_df)

        print("\n--- Scores moyens globaux ---")
        average_scores = results_df.mean(numeric_only=True)
        print(average_scores)

        print("\n--- Scores moyens par catégorie (simple / complex / noisy) ---")
        print(results_df.groupby("category").mean(numeric_only=True))

        print("\n--- Interprétation ---")
        thresholds = {
            "faithfulness": 0.8,
            "answer_relevancy": 0.7,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }
        for metric, threshold in thresholds.items():
            if metric in average_scores:
                score = average_scores[metric]
                status = "✅" if score >= threshold else "⚠️"
                print(f"  {status} {metric}: {score:.2f} (seuil: {threshold})")

        os.makedirs("evaluation/results", exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_df.to_csv(f"evaluation/results/results_{timestamp}.csv", index=False)
        print(f"\nRésultats sauvegardés dans evaluation/results/results_{timestamp}.csv")

    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()