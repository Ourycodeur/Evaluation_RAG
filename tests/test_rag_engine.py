"""
Tests du pipeline RAG (recherche vectorielle) — couvre le cas nominal (chunks
retrouvés) et les cas limites (index vide, aucun résultat pertinent, score
minimum trop élevé, échec API).

Deux catégories de tests :
- Tests MOCKÉS (par défaut) : n'appellent jamais l'API Mistral, rapides et
  gratuits, exécutés à chaque `pytest`. Construisent un index Faiss synthétique
  en mémoire pour tester la logique de vector_store.py de façon déterministe.
- Test D'INTÉGRATION (marqué @pytest.mark.integration) : utilise le vrai index
  Faiss sur disque et un vrai appel à l'API Mistral. Nécessite MISTRAL_API_KEY
  et un index déjà construit (cf. indexer.py). Exclu du run par défaut.
  Lancer explicitement avec : pytest -m integration
"""

import numpy as np
import pytest
import faiss
from unittest.mock import MagicMock

from utils.vector_store import VectorStoreManager


# ============================================================
# FIXTURES — construisent un VectorStoreManager avec un index
# Faiss synthétique en mémoire, sans jamais appeler l'API Mistral.
# ============================================================

@pytest.fixture
def fake_chunks():
    return [
        {
            "id": "0_0",
            "text": "Les Wolves ont progressé offensivement cette saison, Anthony Edwards fait "
                    "davantage confiance à ses coéquipiers.",
            "metadata": {"source": "Reddit_1.pdf", "filename": "Reddit_1.pdf", "category": "reddit"},
        },
        {
            "id": "0_1",
            "text": "Le Magic d'Orlando est critiqué pour son manque de tireurs à trois points "
                    "malgré une bonne défense.",
            "metadata": {"source": "Reddit_1.pdf", "filename": "Reddit_1.pdf", "category": "reddit"},
        },
        {
            "id": "1_0",
            "text": "Shai Gilgeous-Alexander est décrit comme un excellent tireur de lancers-francs, "
                    "autour de 90% de réussite.",
            "metadata": {"source": "Reddit_2.pdf", "filename": "Reddit_2.pdf", "category": "reddit"},
        },
    ]


def _fake_embedding(text: str, dim: int = 8) -> np.ndarray:
    """Génère un embedding déterministe à partir du texte (hash positionnel),
    pour simuler des vecteurs stables sans dépendre d'un vrai modèle. Suffisant
    pour tester la mécanique de recherche, pas la qualité sémantique réelle."""
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    return rng.rand(dim).astype("float32")


@pytest.fixture
def populated_vector_store(fake_chunks, monkeypatch):
    """VectorStoreManager avec un index Faiss réellement construit à partir de
    fake_chunks, et l'appel API Mistral mocké (aucun accès réseau)."""

    # Empêche le __init__ réel de charger un index depuis le disque
    monkeypatch.setattr(VectorStoreManager, "_load_index_and_chunks", lambda self: None)

    store = VectorStoreManager()
    store.mistral_client = MagicMock()

    # Index Faiss réel (pas mocké), construit à partir d'embeddings déterministes
    embeddings = np.array([_fake_embedding(c["text"]) for c in fake_chunks]).astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    store.index = index
    store.document_chunks = fake_chunks

    def _fake_embeddings_call(model, input):
        query_text = input[0]
        if "hors-sujet" in query_text:
            vec = np.random.RandomState(999).rand(8).astype("float32")
        else:
            vec = _fake_embedding(fake_chunks[0]["text"])  # proche du chunk "Wolves"
        fake_response = MagicMock()
        fake_response.data = [MagicMock(embedding=vec.tolist())]
        return fake_response

    store.mistral_client.embeddings.side_effect = _fake_embeddings_call
    return store


@pytest.fixture
def empty_vector_store(monkeypatch):
    """VectorStoreManager sans index chargé (simule l'absence d'indexation)."""
    monkeypatch.setattr(VectorStoreManager, "_load_index_and_chunks", lambda self: None)
    store = VectorStoreManager()
    store.mistral_client = MagicMock()
    store.index = None
    store.document_chunks = []
    return store


# ============================================================
# TESTS MOCKÉS — cas nominal : des chunks sont bien récupérés
# ============================================================

class TestRecuperationChunks:

    def test_search_retourne_des_chunks_pour_une_requete_pertinente(self, populated_vector_store):
        """Cas nominal : une requête proche sémantiquement d'un chunk indexé
        doit retourner au moins un résultat, avec la structure attendue."""
        results = populated_vector_store.search("Comment jouent les Wolves cette saison ?", k=3)

        assert len(results) > 0, "Aucun chunk retourné alors qu'un chunk pertinent existe dans l'index"
        assert results[0]["text"] in [c["text"] for c in populated_vector_store.document_chunks]
        assert "score" in results[0]
        assert "metadata" in results[0]
        assert results[0]["metadata"]["source"] == "Reddit_1.pdf"

    def test_search_respecte_le_parametre_k(self, populated_vector_store):
        """Le nombre de résultats retournés ne doit jamais dépasser k."""
        results = populated_vector_store.search("Question quelconque", k=2)
        assert len(results) <= 2

    def test_search_resultats_tries_par_score_decroissant(self, populated_vector_store):
        results = populated_vector_store.search("Question quelconque", k=3)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Résultats non triés par pertinence décroissante"

    def test_search_avec_min_score_filtre_les_resultats_peu_pertinents(self, populated_vector_store):
        """Un score minimum très élevé doit exclure les chunks peu pertinents,
        potentiellement jusqu'à ne plus rien retourner."""
        results_sans_filtre = populated_vector_store.search("Question quelconque hors-sujet", k=3)
        results_avec_filtre_strict = populated_vector_store.search(
            "Question quelconque hors-sujet", k=3, min_score=0.99
        )
        assert len(results_avec_filtre_strict) <= len(results_sans_filtre)


# ============================================================
# TESTS MOCKÉS — cas limites : AUCUN chunk ne doit être retourné
# ============================================================

class TestAbsenceDeChunks:

    def test_search_retourne_liste_vide_si_index_non_charge(self, empty_vector_store):
        """Si l'index Faiss n'a jamais été construit/chargé (cas du bug corrigé :
        chemin de config invalide, indexer.py jamais lancé), search() doit
        retourner une liste vide, PAS lever d'exception."""
        results = empty_vector_store.search("N'importe quelle question", k=5)
        assert results == []

    def test_search_ne_leve_pas_d_exception_si_index_vide(self, empty_vector_store):
        """Garde-fou : même avec des paramètres inhabituels, aucune exception ne
        doit remonter (rag_pipeline.ask_rag dépend de ce comportement)."""
        try:
            results = empty_vector_store.search("", k=0)
        except Exception as e:
            pytest.fail(f"search() a levé une exception au lieu de retourner [] : {e}")
        assert results == []

    def test_search_gere_echec_api_embedding_sans_crash(self, populated_vector_store):
        """Si l'appel Mistral pour générer l'embedding de la requête échoue
        (rate limit, réseau), search() doit retourner [] plutôt que de crasher
        toute la chaîne au-dessus (rag_pipeline.ask_rag)."""
        populated_vector_store.mistral_client.embeddings.side_effect = Exception("Erreur API simulée")
        results = populated_vector_store.search("Une question", k=3)
        assert results == []


# ============================================================
# TEST D'INTÉGRATION — utilise le VRAI index Faiss + la VRAIE API Mistral.
# Exclu par défaut. Lancer explicitement avec : pytest -m integration
# Nécessite MISTRAL_API_KEY valide + index déjà construit (indexer.py exécuté).
# ============================================================

@pytest.mark.integration
class TestIntegrationReelle:

    def test_recherche_reelle_retourne_des_chunks(self):
        """Test de bout en bout SANS mock : vérifie que le vrai index sur disque
        contient des données et que l'API Mistral répond. Sert de garde-fou
        avant une démo/soutenance : si ça échoue, c'est un problème réel
        d'index/clé API, pas un problème de logique de code."""
        store = VectorStoreManager()

        assert store.index is not None, (
            "Aucun index chargé — vérifie que indexer.py a tourné et que "
            "FAISS_INDEX_FILE dans config.py pointe vers le bon dossier."
        )
        assert store.index.ntotal > 0, "L'index existe mais ne contient aucun vecteur."

        results = store.search("Que pensent les fans des Timberwolves ?", k=5)
        assert len(results) > 0, (
            "Aucun résultat retourné — index vide, corpus ne couvrant pas ce "
            "sujet, ou échec API silencieux (vérifier les logs)."
        )
        for r in results:
            assert r["text"]
            assert 0 <= r["score"] <= 100
            
from unittest.mock import patch

from utils.rag_pipeline import ask_rag


@patch("utils.rag_pipeline.generer_reponse")
@patch("utils.rag_pipeline.vector_store_manager")
def test_ask_rag_retourne_answer_et_contexts(
    mock_vector_store,
    mock_generate
):

    mock_vector_store.search.return_value = [
        {
            "text": "Anthony Edwards est le leader offensif.",
            "metadata": {
                "source": "Reddit.pdf"
            },
            "score": 95
        }
    ]

    mock_generate.return_value = (
        "Anthony Edwards est le leader offensif."
    )

    result = ask_rag(
        "Qui est le leader offensif ?"
    )

    assert "answer" in result
    assert "contexts" in result

    assert len(result["contexts"]) == 1