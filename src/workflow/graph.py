from langgraph.graph import StateGraph, START, END
from src.workflow.state import RiskAnalysisState

# Import each node from its correct domain package
from src.stage1_extract_classify.nodes import node_extract_clauses
from src.stage3_risk_agent.nodes import (
    node_faiss_embedding,
    node_risk_classifier,
    node_mistral_explainer,
)
from src.stage4_report_gen.nodes import node_report_generation


def build_workflow():
    """
    Constructs the full E2E LangGraph DAG for the Legal Contract Risk Analyzer.

    Pipeline flow:
        START
          ├── Node A (Stage 1+2): DeBERTa clause extraction     ──► Node C
          └── Node B (Stage 3):   FAISS vector indexing  ────────────────────┐
                                                                              ▼
                                                Node C (Stage 3): Risk scoring ──► Node D
                                                Node D (Stage 3): Mistral RAG  ──► Node E
                                                Node E (Stage 4): Report Gen   ──► END
    """
    builder = StateGraph(RiskAnalysisState)

    # --- Register nodes by their stage domain ---
    builder.add_node("Node_A_Stage1_Extract", node_extract_clauses)
    builder.add_node("Node_B_Stage3_FAISS", node_faiss_embedding)
    builder.add_node("Node_C_Stage3_RiskClassify", node_risk_classifier)
    builder.add_node("Node_D_Stage3_MistralRAG", node_mistral_explainer)
    builder.add_node("Node_E_Stage4_ReportGen", node_report_generation)

    # --- Edge Routing ---
    # 1. PARALLEL LAUNCH: Both extraction and FAISS indexing start immediately on upload
    builder.add_edge(START, "Node_A_Stage1_Extract")
    builder.add_edge(START, "Node_B_Stage3_FAISS")

    # 2. SEQUENCE: Risk classifier waits for DeBERTa extraction to complete
    builder.add_edge("Node_A_Stage1_Extract", "Node_C_Stage3_RiskClassify")

    # 3. CONVERGENCE: Mistral RAG waits for BOTH the risk flags (C) AND the FAISS store (B)
    builder.add_edge("Node_C_Stage3_RiskClassify", "Node_D_Stage3_MistralRAG")
    builder.add_edge("Node_B_Stage3_FAISS", "Node_D_Stage3_MistralRAG")

    # 4. Report generation waits for Mistral explanations
    builder.add_edge("Node_D_Stage3_MistralRAG", "Node_E_Stage4_ReportGen")

    # 5. Terminate
    builder.add_edge("Node_E_Stage4_ReportGen", END)

    return builder.compile()


risk_graph = build_workflow()


if __name__ == "__main__":
    print("\n========================================")
    print("  E2E PIPELINE ARCHITECTURE FLOW")
    print("========================================")
    print(risk_graph.get_graph().draw_ascii())
    print("========================================\n")
