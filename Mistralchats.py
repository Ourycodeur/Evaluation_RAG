# MistralChat.py (version RAG + Tool SQL, Étape 2)
import streamlit as st
import logging

from utils.config import APP_TITLE, NAME, MODEL_NAME
from utils.vector_store import VectorStoreManager
from tools.agent_router import rag_agent, RagAgentDeps
from utils.Reporting import log_report, Timer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


@st.cache_resource
def get_vector_store_manager():
    logging.info("Tentative de chargement du VectorStoreManager...")
    try:
        manager = VectorStoreManager()
        if manager.index is None or not manager.document_chunks:
            st.error("L'index vectoriel ou les chunks n'ont pas pu être chargés.")
            st.warning("Assurez-vous d'avoir exécuté 'python indexer.py' après avoir placé vos fichiers dans le dossier 'inputs'.")
            logging.error("Index Faiss ou chunks non trouvés/chargés par VectorStoreManager.")
            return None
        logging.info(f"VectorStoreManager chargé avec succès ({manager.index.ntotal} vecteurs).")
        return manager
    except Exception as e:
        st.error(f"Erreur inattendue lors du chargement du VectorStoreManager: {e}")
        logging.exception("Erreur chargement VectorStoreManager")
        return None


vector_store_manager = get_vector_store_manager()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": f"Bonjour ! Je suis votre analyste IA pour la {NAME}. Posez-moi vos questions sur les "
                   f"équipes, les joueurs, les statistiques ou l'avis des fans, et je choisirai la meilleure "
                   f"source pour vous répondre."
    }]


def generer_reponse(prompt: str) -> tuple[str, RagAgentDeps]:
    """
    Envoie la question à l'agent (qui décide lui-même d'appeler search_nba_context
    et/ou query_nba_stats). Retourne la réponse ET les deps (pour la journalisation
    et un éventuel affichage des sources dans l'UI).
    """
    deps = RagAgentDeps(vector_store=vector_store_manager, search_k=5, retrieved_contexts=[], sql_tool_calls=[])

    if not prompt:
        logging.warning("Tentative de génération de réponse avec un prompt vide.")
        return "Je ne peux pas traiter une demande vide.", deps

    try:
        logging.info(f"Appel de l'agent RAG pour la question : '{prompt}'")
        result = rag_agent.run_sync(prompt, deps=deps)
        return result.output, deps
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'agent : {e}")
        logging.exception("Erreur agent pendant generer_reponse")
        return "Je suis désolé, une erreur technique m'empêche de répondre. Veuillez réessayer plus tard.", deps


# --- Interface Utilisateur Streamlit ---
st.title(APP_TITLE)
st.caption(f"Assistant virtuel pour {NAME} | Modèle: {MODEL_NAME}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input(f"Posez votre question sur la {NAME}..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if vector_store_manager is None:
        st.error("Le service de recherche de connaissances n'est pas disponible. Impossible de traiter votre demande.")
        logging.error("VectorStoreManager non disponible.")
        st.stop()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.text("...")

        status = "success"
        error_message = None
        with Timer() as timer:
            try:
                response_content, deps = generer_reponse(prompt)
            except Exception as e:
                status = "error"
                error_message = str(e)
                response_content = "Erreur technique, réessayez plus tard."
                deps = RagAgentDeps(vector_store=vector_store_manager)

        message_placeholder.write(response_content)

        # Traçabilité : journalisation dans la table `reports` (question, SQL généré
        # le cas échéant, réponse finale, statut, latence) — cf. brief traçabilité mission.
        log_report(
            user_question=prompt,
            deps=deps,
            final_answer=response_content,
            status=status,
            error_message=error_message,
            latency_ms=timer.elapsed_ms,
        )

        # Petite transparence pour l'utilisateur : indique si des stats SQL ont été utilisées
        if deps.sql_tool_calls:
            with st.expander("🔍 Détails techniques (SQL utilisé)"):
                for call in deps.sql_tool_calls:
                    st.code(call.get("sql") or "Aucune requête générée", language="sql")

    st.session_state.messages.append({"role": "assistant", "content": response_content})

st.markdown("---")
st.caption("Powered by Mistral AI, Faiss & PostgreSQL/SQLite | Data-driven NBA Insights")