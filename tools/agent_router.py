# utils/agent_router.py
"""
Agent RAG unifié (Étape 2) — utilisé À LA FOIS par l'application Streamlit
(MistralChat.py) ET par l'évaluation Ragas (rag_engine.py), pour garantir que
ce qui est évalué est exactement ce qui tourne en production.

CHOIX TECHNIQUE IMPORTANT :
pydantic-ai nécessite la nouvelle génération du SDK `mistralai` (classe
`Mistral`) pour son provider natif "mistral:...", ce qui casse `MistralClient`
utilisé partout ailleurs dans le projet (vector_store.py notamment) — cf.
diagnostic détaillé échangé sur ce point. Pour éviter de migrer tout le
pipeline d'embeddings, on pilote Mistral via le provider OpenAI générique de
pydantic-ai, pointé sur l'endpoint HTTP compatible OpenAI de Mistral. Ça évite
totalement d'importer le package `mistralai` ici : aucun conflit de version
possible avec vector_store.py.

⚠️ Le nom de la classe modèle a changé entre versions de pydantic-ai
(`OpenAIModel` -> `OpenAIChatModel`). Le bloc d'import ci-dessous gère les deux
cas pour éviter un nouveau piège de breaking change de dépendance.
"""

import logging
from dataclasses import dataclass, field

# Chargement explicite de la config AVANT toute construction d'objet dépendant
# de MISTRAL_API_KEY, pour éviter le bug d'ordre d'import repéré dans
# MistralChat.py (l'agent ne doit jamais être instancié avant que .env soit chargé).
from utils.config import MISTRAL_API_KEY, MODEL_NAME, STATS_DATABASE_URL
from utils.vector_store import VectorStoreManager
from utils.tool_sql import run_sql_tool

from pydantic_ai import Agent, RunContext
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings

try:
    from pydantic_ai.models.openai import OpenAIChatModel as _OpenAIModelClass
except ImportError:
    from pydantic_ai.models.openai import OpenAIModel as _OpenAIModelClass  # anciennes versions

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if not MISTRAL_API_KEY:
    logging.error("MISTRAL_API_KEY manquante : l'agent RAG ne pourra pas appeler le LLM.")


@dataclass
class RagAgentDeps:
    """Dépendances injectées dans l'agent (accessibles aux tools via RunContext)."""
    vector_store: VectorStoreManager
    search_k: int = 5
    # Rempli par le tool pendant l'exécution : permet de récupérer après coup les chunks
    # bruts effectivement utilisés (nécessaire pour les métriques Ragas context_precision/
    # context_recall, puisque l'agent décide lui-même s'il appelle le tool ou non).
    retrieved_contexts: list = field(default_factory=list)
    # Journal des appels au tool SQL pendant l'exécution (question, SQL généré, statut) ;
    # destiné à alimenter la table `reports` pour la traçabilité (cf. brief général mission).
    sql_tool_calls: list = field(default_factory=list)


SYSTEM_PROMPT = """Tu es 'NBA Analyst AI', un assistant expert sur la ligue de basketball NBA.

Tu as accès à DEUX outils :

1. search_nba_context : recherche dans une base de discussions de fans (threads Reddit r/nba).
   À utiliser pour toute question qualitative, d'opinion, de contexte narratif ou de débat
   sur des joueurs/équipes (ex: "que pensent les fans de X ?", "pourquoi Y est critiqué ?").

2. query_nba_stats : interroge une base de données statistiques structurée (joueurs, équipes,
   stats de la saison). À utiliser pour TOUTE question chiffrée, comparative ou classante
   (ex: "qui a marqué le plus de points ?", "quel est le % à 3 points de X ?", "quelle équipe
   a la meilleure défense ?"). Ne devine JAMAIS une statistique : appelle systématiquement cet
   outil dès qu'un nombre, un classement ou une comparaison chiffrée est demandé.

Règles impératives :
1. Réponds UNIQUEMENT à partir des informations retournées par tes outils. N'invente jamais
   une information que tes outils n'ont pas fournie.
2. Si aucune information pertinente n'est trouvée par tes outils, réponds explicitement
   "Je ne dispose pas de cette information dans mes sources" plutôt que de deviner.
3. Le contexte retourné par search_nba_context provient d'opinions de fans sur des forums,
   pas de faits établis. Restitue les opinions comme telles ("certains utilisateurs pensent
   que...") sans les présenter comme des faits objectifs.
4. Si plusieurs avis contradictoires apparaissent, mentionne cette divergence plutôt que de
   n'en retenir qu'un seul.
5. Reste attentif au ton ironique ou sarcastique dans les commentaires de forum : ne prends
   pas au premier degré une remarque manifestement moqueuse ou exagérée.
6. Si une question mélange les deux besoins (ex: "qui est le meilleur marqueur et que pensent
   les fans de lui ?"), appelle les deux outils et synthétise les deux réponses.
7. Si query_nba_stats retourne une erreur ou un statut d'échec, ne tente pas de deviner la
   réponse : indique que la donnée n'a pas pu être récupérée.
"""

_model = _OpenAIModelClass(
    MODEL_NAME,
    provider=OpenAIProvider(base_url="https://api.mistral.ai/v1", api_key=MISTRAL_API_KEY),
)

rag_agent = Agent(
    _model,
    deps_type=RagAgentDeps,
    system_prompt=SYSTEM_PROMPT,
    model_settings=ModelSettings(temperature=0.2, max_tokens=800),
)


@rag_agent.tool
def search_nba_context(ctx: RunContext[RagAgentDeps], query: str) -> str:
    """
    Recherche des informations pertinentes dans la base de connaissances NBA
    (discussions de fans, threads Reddit) à partir d'une requête sémantique.

    À utiliser pour toute question qualitative (avis, discussions, contexte
    narratif, opinions sur des joueurs/équipes). Ne pas utiliser pour des
    statistiques chiffrées précises (non couvertes par cette base).

    Args:
        query: la requête de recherche, reformulée si besoin à partir de la
            question de l'utilisateur pour maximiser la pertinence sémantique.
    """
    logging.info(f"[tool:search_nba_context] query='{query}'")
    try:
        results = ctx.deps.vector_store.search(query, k=ctx.deps.search_k)
    except Exception as e:
        logging.exception("Erreur pendant l'appel à vector_store.search dans le tool")
        return f"Erreur technique lors de la recherche : {e}"

    if not results:
        logging.warning(f"[tool:search_nba_context] aucun résultat pour '{query}'")
        return "Aucun document pertinent trouvé dans la base de connaissances."

    # Capture des chunks bruts pour l'évaluation Ragas (context_precision/context_recall)
    ctx.deps.retrieved_contexts.extend(r["text"] for r in results)

    formatted = "\n\n".join(
        f"[Source {i + 1} - {r['metadata'].get('source', 'inconnue')} "
        f"(score: {r['score']:.1f}%)]\n{r['text']}"
        for i, r in enumerate(results)
    )
    logging.info(f"[tool:search_nba_context] {len(results)} résultat(s) retourné(s)")
    return formatted


@rag_agent.tool
def query_nba_stats(ctx: RunContext[RagAgentDeps], question: str) -> str:
    """
    Interroge la base de données statistiques NBA (joueurs, équipes, stats saison)
    pour répondre à une question chiffrée, comparative ou classante.

    À utiliser pour tout ce qui implique un nombre, un classement, une comparaison
    statistique entre joueurs/équipes. Ne pas utiliser pour des questions d'opinion.

    Args:
        question: la question de l'utilisateur, éventuellement reformulée pour être
            sans ambiguïté (ex: préciser "sur la saison" si la question est vague).
    """
    logging.info(f"[tool:query_nba_stats] question='{question}'")
    result = run_sql_tool(question, database_url=STATS_DATABASE_URL)

    # Traçabilité : chaque appel est loggé dans deps pour être persisté en table `reports`
    # par la couche appelante (Streamlit / script d'éval), cf. db/models_db.py::Report
    ctx.deps.sql_tool_calls.append({
        "question": question,
        "sql": result.get("sql"),
        "status": result["status"],
        "error": result.get("error"),
    })

    if result["status"] == "error":
        logging.warning(f"[tool:query_nba_stats] échec: {result['error']}")
        return f"Erreur lors de l'interrogation des statistiques : {result['error']}"

    return f"Requête SQL exécutée : {result['sql']}\nRésultat : {result['result']}" 