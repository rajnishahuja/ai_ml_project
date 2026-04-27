from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from src.workflow.state import RiskAnalysisState

# Import each node from its correct domain package
from src.stage1_extract_classify.nodes import node_extract_clauses
from src.stage3_risk_agent.nodes import (
    node_risk_classifier,
    node_mistral_router,
)
from src.stage4_report_gen.nodes import node_report_generation


def continue_to_mistral(state: RiskAnalysisState):
    """
    Dispatcher logic for the Map-Reduce fan-out.
    Takes the flagged clauses from Node C and creates one parallel branch
    (Send) per clause routed to the Mistral Router (Node D).

    Each Send payload carries:
      - clause_data        : the single clause this worker assesses
      - extracted_clauses  : ALL clauses in the contract — used by the worker
                             to bind contract_search via closure so the
                             tool sees same-document context without Mistral
                             having to transport it
      - document_id        : contract identifier (sanity arg for tools)
      - metadata_block     : Parties / Effective Date / Expiration Date for
                             direct prompt injection (ARCHITECTURE.md says
                             Parties is the #1 driver of label flips —
                             must be in Mistral's prompt on turn 1)

    Note: passing the full extracted_clauses list in each Send is a Python
    reference, not a deep copy — overhead is negligible for ~13 clauses.
    """
    extracted_clauses = state.get("extracted_clauses", [])
    metadata_block = state.get("metadata_block", {})
    document_id = state.get("document_id", "")

    return [
        Send("Node_D_Mistral_Router", {
            "clause_data": clause,
            "extracted_clauses": extracted_clauses,
            "document_id": document_id,
            "metadata_block": metadata_block,
        })
        for clause in state["flagged_clauses"]
    ]


def build_workflow():
    """
    Constructs the full E2E LangGraph DAG for the Legal Contract Risk Analyzer.

    Pipeline flow:
        START
          ├── Node A (Stage 1+2): DeBERTa clause extraction
          ▼
        Node C (Stage 3): Risk scoring (Dispatcher)
          ▼ (Map-Reduce Fan-Out)
        Node D (Stage 3): Mistral RAG Router (Workers running in parallel)
          ▼ (Map-Reduce Fan-In)
        Node E (Stage 4): Report Gen   ──► END
    """
    builder = StateGraph(RiskAnalysisState)

    # --- Register nodes by their stage domain ---
    builder.add_node("Node_A_Stage1_Extract", node_extract_clauses)
    builder.add_node("Node_C_Stage3_RiskClassify", node_risk_classifier)
    builder.add_node("Node_D_Mistral_Router", node_mistral_router)
    builder.add_node("Node_E_Stage4_ReportGen", node_report_generation)

    # --- Edge Routing ---
    builder.add_edge(START, "Node_A_Stage1_Extract")
    builder.add_edge("Node_A_Stage1_Extract", "Node_C_Stage3_RiskClassify")

    # The FAN-OUT Conditional Edge (Dispatcher)
    builder.add_conditional_edges(
        "Node_C_Stage3_RiskClassify",
        continue_to_mistral,
        ["Node_D_Mistral_Router"]
    )

    # The FAN-IN Edge (Worker to Aggregator)
    builder.add_edge("Node_D_Mistral_Router", "Node_E_Stage4_ReportGen")

    # Terminate
    builder.add_edge("Node_E_Stage4_ReportGen", END)

    return builder.compile()


risk_graph = build_workflow()


if __name__ == "__main__":
    print("\n========================================")
    print("  E2E PIPELINE ARCHITECTURE FLOW")
    print("========================================")
    print(risk_graph.get_graph().draw_ascii())
    print("========================================\n")
