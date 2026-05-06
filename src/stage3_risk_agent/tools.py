"""
LangGraph tools for the Stage 3 risk assessment agent.

Both tools are created via factory functions that close over their
dependencies (index path, clause list). The returned @tool functions
have clean LLM-facing signatures — the agent calls them with simple
string arguments and gets back JSON-serialisable dicts.

Usage (in agent.py):
    precedent_search = make_precedent_search_tool(cfg["faiss_index_path"])
    contract_search  = make_contract_search_tool(clauses)
    agent = create_react_agent(llm, [precedent_search, contract_search])
"""

import logging
from dataclasses import asdict

from langchain_core.tools import tool

from src.common.schema import ClauseObject
from src.stage3_risk_agent.embeddings import query_index
from src.stage3_risk_agent.risk_classifier import RiskClassifier

logger = logging.getLogger(__name__)


def make_precedent_search_tool(index_path: str):
    """Return a @tool that searches the labeled clause corpus by similarity.

    Args:
        index_path: Path to the FAISS index file (data/faiss_index/clauses.index).
    """

    @tool
    def precedent_search(clause_text: str, k: int = 5,
                         min_similarity: float = 0.75) -> list[dict]:
        """Search the labeled clause corpus for clauses similar to the one you are assessing.

        Returns only clauses with cosine similarity >= min_similarity (default 0.75),
        so results are genuinely relevant — not just the closest available matches.
        If fewer than k results meet the threshold, only those are returned.
        Receiving 0 results means no strong precedents exist in the corpus for this
        clause — in that case, call deberta_classify for a model-based signal instead.

        Args:
            clause_text: Full text of the clause to search for.
            k: Maximum number of results to return (default 5).
            min_similarity: Minimum cosine similarity to include a result (default 0.75).
                            Lower to 0.6 only if 0.75 returns 0 results and you need
                            any available signal.

        Returns:
            List of similar clauses ordered by descending similarity, each with
            clause_type, risk_level, similarity score, and clause text.
            Empty list if no clauses meet the similarity threshold.
        """
        results = query_index(clause_text, index_path, k)
        filtered = [r for r in results if r.similarity >= min_similarity]
        return [asdict(r) for r in filtered]

    return precedent_search


def make_deberta_classify_tool(classifier: RiskClassifier, signing_party: str):
    """Return a @tool that runs DeBERTa risk classification on a clause.

    Args:
        classifier: Loaded RiskClassifier instance (Ens-F).
        signing_party: Pre-resolved signing party for this document.
    """

    @tool
    def deberta_classify(clause_text: str, clause_type: str) -> dict:
        """Get DeBERTa's risk classification for a clause.

        Use this when precedent evidence is weak (low similarity scores or
        split votes) and you want a model-based signal to help resolve
        uncertainty. DeBERTa was fine-tuned on 3,400 labeled CUAD clauses
        and captures clause-type patterns that may not appear in precedents.

        Treat the result as one signal among others — not a final answer.
        It is most reliable when confidence is high (> 0.75) and most
        useful as a tiebreaker when precedents are split.

        Args:
            clause_text: Full text of the clause to classify.
            clause_type: CUAD clause type (e.g. "Cap On Liability").

        Returns:
            dict with keys: label (LOW/MEDIUM/HIGH), confidence (0-1).
        """
        result = classifier.predict(
            clause_text=clause_text,
            clause_type=clause_type,
            signing_party=signing_party,
        )
        return {"label": result["label"], "confidence": round(result["confidence"], 3)}

    return deberta_classify


def make_contract_search_tool(clauses: list[ClauseObject]):
    """Return a @tool that retrieves all other clauses from the same contract.

    Args:
        clauses: All ClauseObjects extracted from the contract by Stage 1+2.
    """

    @tool
    def contract_search(current_clause_id: str) -> list[dict]:
        """Retrieve all other clauses extracted from the same contract.

        Call this when precedent evidence alone is insufficient — specifically when
        party roles or cross-clause context could change the risk assessment. For
        example: an IP assignment clause may look HIGH risk in isolation but be LOW
        risk if the contract shows the signing party is the one receiving rights.
        Seeing the Parties clause, Governing Law, or related restrictions from the
        same document can resolve ambiguity that precedent search cannot.

        Do NOT call this if precedent_search already provided clear consensus — use
        it only to resolve ambiguity that requires same-contract context.

        Args:
            current_clause_id: The clause_id of the clause currently being assessed
                               (excluded from results so you only see sibling clauses).

        Returns:
            List of all other clauses from this contract, each with clause_type
            and clause_text.
        """
        current = next((c for c in clauses if c.clause_id == current_clause_id), None)
        if current is None:
            logger.warning("contract_search: clause_id %s not found", current_clause_id)
            return []
        siblings = [
            {"clause_type": c.clause_type, "clause_text": c.clause_text}
            for c in clauses
            if c.clause_id != current_clause_id and c.document_id == current.document_id
        ]
        logger.debug("contract_search: returning %d sibling clauses for doc %s",
                     len(siblings), current.document_id)
        return siblings

    return contract_search
