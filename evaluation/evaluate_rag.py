# evaluation/evaluate_ragas_step3.py
"""
Script d'évaluation Ragas — ÉTAPE 3 (avec Tool SQL branché sur l'agent).

Utilise le MÊME testset que la baseline Étape 1 (testset_common.py), mais cette
fois via l'agent complet (tools/agent_router.py : search_nba_context +
query_nba_stats), pour un comparatif avant/après valide.

Contrairement à l'Étape 1, les questions chiffrées du TEST_SET (catégories
"simple" et "complex") sont maintenant censées obtenir de BONS scores, puisque
l'agent peut interroger la vraie base de stats via le Tool SQL au lieu de
chercher (en vain) dans les threads Reddit.
"""

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

from tools.agent_router import rag_agent, RagAgentDeps
from utils.vector_store import VectorStoreManager
from utils.data_loader import load_and_parse_files
from utils.reddit_cleaner import clean_text
from utils.config import INPUT_DIR
from testset import TEST_SET

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "")


def ensure_index_built(vector_store, force_rebuild: bool = False) -> None:
    """Identique à l'Étape 1 : construit l'index Faiss si besoin avant l'éval."""
    if vector_store.index is not None and vector_store.document_chunks and not force_rebuild:
        print(f"Index déjà chargé ({vector_store.index.ntotal} vecteurs) — pas de ré-indexation.")
        return

    print(f"Aucun index utilisable trouvé — indexation depuis {INPUT_DIR}...")
    documents = load_and_parse_files(INPUT_DIR)
    if not documents:
        print(f"⚠️ Aucun document trouvé dans {INPUT_DIR}.")
        return

    documents = clean_text(documents)
    print(f"{len(documents)} document(s) chargé(s) et nettoyé(s), construction de l'index en cours...")
    vector_store.build_index(documents)

    if vector_store.index is None:
        print("❌ Échec de la construction de l'index — vérifie les logs ci-dessus.")
    else:
        print(f"✅ Index construit : {vector_store.index.ntotal} vecteurs.")


def collect_contexts(deps: RagAgentDeps) -> list[str]:
    """
    Construit la liste de contextes pour Ragas à partir de TOUT ce que l'agent a
    utilisé pendant le run : chunks Reddit (search_nba_context) ET requêtes SQL
    exécutées (query_nba_stats). Pour une question chiffrée répondue via SQL, le
    "contexte" qui justifie la réponse est la requête + son résultat — pas un
    chunk de texte — donc on le formate comme tel pour que faithfulness/
    context_precision aient quelque chose de concret à évaluer.
    """
    contexts = list(deps.retrieved_contexts)
    for call in deps.sql_tool_calls:
        if call.get("status") == "success":
            contexts.append(
                f"[Requête SQL exécutée] {call.get('sql')}\n[Résultat] {call.get('result')}"
            )
    if not contexts:
        contexts = ["Aucun contexte récupéré (ni recherche vectorielle, ni requête SQL exécutée avec succès)."]
    return contexts


def ask_rag_with_tools(question: str, vector_store: VectorStoreManager, k: int = 5) -> dict:
    """Enveloppe autour de l'agent complet (RAG + Tool SQL)."""
    deps = RagAgentDeps(vector_store=vector_store, search_k=k, retrieved_contexts=[], sql_tool_calls=[])
    try:
        result = rag_agent.run_sync(question, deps=deps)
        answer = result.output
    except Exception as e:
        print(f"  ⚠️ Erreur agent pour la question '{question}': {e}")
        answer = "Erreur: impossible de générer une réponse."

    return {
        "answer": answer,
        "contexts": collect_contexts(deps),
        "sql_used": bool(deps.sql_tool_calls),
        "rag_used": bool(deps.retrieved_contexts),
    }


def main():
    if not MISTRAL_API_KEY:
        print("⚠️ AVERTISSEMENT : Clé API Mistral non trouvée (MISTRAL_API_KEY).")

    vector_store = VectorStoreManager()
    ensure_index_built(vector_store)

    questions, answers, contexts, ground_truths, categories, tools_used = [], [], [], [], [], []

    for i, sample in enumerate(TEST_SET):
        print(f"\n[{i + 1}/{len(TEST_SET)}] Traitement : {sample['question']}")

        rag_result = ask_rag_with_tools(sample["question"], vector_store=vector_store)

        questions.append(sample["question"])
        answers.append(rag_result["answer"])
        contexts.append(rag_result["contexts"])
        ground_truths.append(sample["ground_truth"])
        categories.append(sample["category"])
        tool_label = []
        if rag_result["sql_used"]:
            tool_label.append("SQL")
        if rag_result["rag_used"]:
            tool_label.append("RAG")
        tools_used.append("+".join(tool_label) if tool_label else "aucun")

        print(f"  → Outil(s) utilisé(s)         : {tools_used[-1]}")
        print(f"  → Contexte récupéré (extrait) : {rag_result['contexts'][0][:150]}...")
        print(f"  → Réponse générée (extrait)   : {rag_result['answer'][:150]}...")

        time.sleep(10)  # anti rate-limit Mistral

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    try:
        print("\nInitialisation LLM et Embeddings Mistral (juge Ragas)...")
        from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

        mistral_llm = ChatMistralAI(mistral_api_key=MISTRAL_API_KEY, model="mistral-large-latest", temperature=0.1)
        mistral_embeddings = MistralAIEmbeddings(mistral_api_key=MISTRAL_API_KEY)

        metrics_to_evaluate = [faithfulness, answer_relevancy, context_precision, context_recall]
        print(f"Métriques : {[m.name for m in metrics_to_evaluate]}")
        print("\nLancement évaluation Ragas (Étape 3 — avec Tool SQL)...")

        results = evaluate(
            dataset=dataset,
            metrics=metrics_to_evaluate,
            llm=mistral_llm,
            embeddings=mistral_embeddings,
            run_config=RunConfig(max_workers=1, timeout=600, max_retries=13, max_wait=120, seed=42),
            raise_exceptions=False,
        )
        print("\n--- Évaluation Ragas terminée ---")

        results_df = results.to_pandas()
        results_df["category"] = categories
        results_df["tools_used"] = tools_used

        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        pd.set_option("display.max_colwidth", 150)

        print("\n--- Résultats par question (Étape 3) ---")
        print(results_df)

        average_scores = results_df.mean(numeric_only=True)
        print("\n--- Scores moyens globaux (Étape 3) ---")
        print(average_scores)

        print("\n--- Scores moyens par catégorie ---")
        print(results_df.groupby("category").mean(numeric_only=True))

        print("\n--- Interprétation ---")
        thresholds = {"faithfulness": 0.8, "answer_relevancy": 0.7, "context_precision": 0.7, "context_recall": 0.6}
        for metric, threshold in thresholds.items():
            if metric in average_scores:
                score = average_scores[metric]
                status = "✅" if score >= threshold else "⚠️"
                print(f"  {status} {metric}: {score:.2f} (seuil: {threshold})")

        results_dir = ROOT_DIR / "evaluation" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = results_dir / f"results_step3_{timestamp}.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nRésultats sauvegardés dans {csv_path}")

        # --- Comparatif avant/après (baseline Étape 1 saisie manuellement, cf. note ci-dessous) ---
        print_comparison(average_scores)

    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        traceback.print_exc()


# ============================================================
# COMPARATIF AVANT/APRÈS
# ============================================================
# Ta baseline Étape 1 n'a pas été sauvegardée en CSV (juste affichée au terminal).
# Remplis ces valeurs avec les scores moyens que tu as vus s'afficher à ce moment-là
# (section "Scores moyens globaux" de ton run evaluate_ragas.py), puis relance ce
# script : la comparaison sera calculée et affichée automatiquement à la fin.
BASELINE_ETAPE_1_SCORES = {
    "faithfulness": 0.17,       # <-- remplace None par le score vu au terminal (ex: 0.42)
    "answer_relevancy": None,
    "context_precision": 0.67,
    "context_recall": 0.67,
}


def print_comparison(step3_scores) -> None:
    print("\n" + "=" * 60)
    print("COMPARATIF ÉTAPE 1 (RAG seul) vs ÉTAPE 3 (RAG + Tool SQL)")
    print("=" * 60)

    if all(v is None for v in BASELINE_ETAPE_1_SCORES.values()):
        print("⚠️ BASELINE_ETAPE_1_SCORES n'est pas rempli (tous les scores sont None).")
        print("   Édite ce fichier avec les scores moyens affichés lors du run Étape 1,")
        print("   puis relance ce script pour voir le comparatif complet.")
        return

    print(f"{'Métrique':<20}{'Étape 1':>12}{'Étape 3':>12}{'Delta':>12}")
    for metric, baseline in BASELINE_ETAPE_1_SCORES.items():
        after = step3_scores.get(metric)
        if baseline is None or after is None:
            print(f"{metric:<20}{'N/A':>12}{after if after is not None else 'N/A':>12}{'N/A':>12}")
            continue
        delta = after - baseline
        sign = "+" if delta >= 0 else ""
        print(f"{metric:<20}{baseline:>12.2f}{after:>12.2f}{sign}{delta:>11.2f}")


if __name__ == "__main__":
    main()