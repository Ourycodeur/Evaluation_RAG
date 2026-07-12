# rag_engine.py
import logging
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from utils.config import (
    MISTRAL_API_KEY,
    MODEL_NAME,
    SEARCH_K
)

from utils.vector_store import VectorStoreManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

client = MistralClient(api_key=MISTRAL_API_KEY)

vector_store_manager = VectorStoreManager()


SYSTEM_PROMPT = """
Tu es 'NBA Analyst AI', un assistant expert sur la ligue NBA.

Réponds UNIQUEMENT à partir du contexte fourni ci-dessous. Si le contexte ne contient
pas l'information nécessaire pour répondre, dis-le explicitement ("Je ne dispose pas
de cette information dans mes sources") plutôt que d'inventer une réponse.

---
{context_str}
---

QUESTION:
{question}

RÉPONSE:
"""


def generer_reponse(messages_for_api):
    """Appelle l'API Mistral. Retourne un message d'erreur lisible plutôt que de
    laisser planter l'appelant (important en boucle d'évaluation : une question qui
    échoue à cause d'un rate limit ne doit pas faire perdre tout le run)."""
    try:
        response = client.chat(
            model=MODEL_NAME,
            messages=messages_for_api,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Erreur lors de l'appel à l'API Mistral (chat) : {e}")
        return "Erreur: impossible de générer une réponse."


def ask_rag(question: str, k: int = None) -> dict:
    k = k or SEARCH_K

    try:
        search_results = vector_store_manager.search(question, k=k)
    except Exception as e:
        logging.error(f"Erreur lors de la recherche vectorielle pour '{question}': {e}")
        search_results = []

    if not search_results:
        logging.warning(f"Aucun contexte trouvé pour la question: '{question}'")
        context_str = "Aucun document pertinent trouvé dans le corpus."
        contexts = ["Aucun document pertinent trouvé dans le corpus."]
    else:
        contexts = [result["text"] for result in search_results]
        context_str = "\n\n---\n\n".join(
            f"Source: {res['metadata'].get('source', '')}\n{res['text']}"
            for res in search_results
        )

    final_prompt = SYSTEM_PROMPT.format(
        context_str=context_str,
        question=question
    )

    messages_for_api = [
        ChatMessage(role="user", content=final_prompt)
    ]

    answer = generer_reponse(messages_for_api)

    return {
        "answer": answer,
        "contexts": contexts
    }