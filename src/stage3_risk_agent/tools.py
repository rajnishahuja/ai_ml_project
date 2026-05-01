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

logger = logging.getLogger(__name__)


def make_precedent_search_tool(index_path: str):
    """Return a @tool that searches the labeled clause corpus by similarity.

    Args:
        index_path: Path to the FAISS index file (data/faiss_index/clauses.index).
    """

    @tool
    def precedent_search(clause_text: str, k: int = 5) -> list[dict]:
        """Search the labeled clause corpus for clauses similar to the one you are assessing.

        Call this first when the risk level is uncertain. Returns the top-k most
        similar clauses from real contracts, each with a verified risk label. Use
        the distribution of labels and the clause context to reason about the
        appropriate risk level for the clause under review.

        Args:
            clause_text: Full text of the clause to search for.
            k: Number of similar clauses to return (default 5).

        Returns:
            List of similar clauses, each with clause_type, risk_level, similarity score,
            and the clause text. Ordered by descending similarity.
        """
        results = query_index(clause_text, index_path, k)
        return [asdict(r) for r in results]

    return precedent_search


def make_contract_search_tool(clauses: list[ClauseObject]):
    """Return a @tool that retrieves all other clauses from the same contract.

    Args:
        clauses: All ClauseObjects extracted from the contract by Stage 1+2.
    """

    @tool
    def contract_search(document_id: str) -> list[dict]:
        """Retrieve all clauses extracted from the same contract.

        Call this when precedent evidence alone is insufficient — specifically when
        party roles or cross-clause context could change the risk assessment. For
        example: an IP assignment clause may look HIGH risk in isolation but be LOW
        risk if the contract shows the signing party is the one receiving rights.

        Do NOT call this if precedent_search already provided clear consensus — use
        it only to resolve ambiguity that requires same-contract context.

        Args:
            document_id: The contract's document identifier.

        Returns:
            List of clauses from the same contract, each with clause_type and
            clause_text. Excludes the clause currently being assessed.
        """
        siblings = [
            {"clause_type": c.clause_type, "clause_text": c.clause_text}
            for c in clauses
            if c.document_id == document_id
        ]
        if not siblings:
            logger.warning("contract_search: no clauses found for document_id=%s", document_id)
        return siblings

    return contract_search
