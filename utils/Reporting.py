# utils/reporting.py
"""
Journalisation des interactions de l'agent dans la table `reports` (traçabilité
demandée dans le cadrage général de la mission). Découplé de rag_agent.py pour
pouvoir être appelé aussi bien depuis MistralChat.py que depuis un futur script
d'évaluation Étape 3, sans dépendance circulaire.
"""

import logging
import time
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils.db_models import Base, Report
from utils.config import STATS_DATABASE_URL

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

_engine = create_engine(STATS_DATABASE_URL)
Base.metadata.create_all(_engine)  # no-op si les tables existent déjà
_Session = sessionmaker(bind=_engine)


def log_report(user_question: str, deps, final_answer: str, status: str = "success", error_message: str = None, latency_ms: int = None) -> None:
    """
    Persiste une interaction dans la table `reports`.

    - generated_sql / sql_result_summary : dérivés de deps.sql_tool_calls si le
      tool SQL a été appelé pendant ce run (peut être vide si seule la recherche
      vectorielle a été utilisée).
    """
    generated_sql = None
    sql_result_summary = None
    if getattr(deps, "sql_tool_calls", None):
        last_call = deps.sql_tool_calls[-1]
        generated_sql = last_call.get("sql")
        sql_result_summary = f"status={last_call.get('status')}, error={last_call.get('error')}"

    session = _Session()
    try:
        report = Report(
            user_question=user_question,
            generated_sql=generated_sql,
            sql_result_summary=sql_result_summary,
            final_answer=final_answer,
            execution_status=status,
            error_message=error_message,
            latency_ms=latency_ms,
        )
        session.add(report)
        session.commit()
    except Exception as e:
        session.rollback()
        logging.error(f"Échec de la journalisation dans `reports` : {e}")
    finally:
        session.close()


class Timer:
    """Petit utilitaire pour mesurer la latence d'un run (ms), utilisé avec log_report."""
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)