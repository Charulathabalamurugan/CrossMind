"""
LangGraph Orchestrator - Manages nodes & iteration loops with adaptive routing
Implements Neural-Symbolic Feedback Loop for hallucination prevention.
"""
from typing import Dict, List, Any, Optional, TypedDict
import json
import logging
from enum import Enum

try:
    from langgraph.graph import StateGraph, END
except Exception:
    END = "END"

    class StateGraph:
        def __init__(self, state_type=None):
            self.nodes = {}
            self.edges = {}
            self.conditional_edges = {}
            self.entry_point = None

        def add_node(self, name, func):
            self.nodes[name] = func

        def set_entry_point(self, name):
            self.entry_point = name

        def add_edge(self, source, target):
            self.edges[source] = target

        def add_conditional_edges(self, source, router, mapping):
            self.conditional_edges[source] = (router, mapping)

        def compile(self):
            return self

        def invoke(self, state):
            current = self.entry_point
            while current and current != END:
                func = self.nodes.get(current)
                if not func:
                    break
                state = func(state)
                if current in self.conditional_edges:
                    router, mapping = self.conditional_edges[current]
                    route_key = router(state)
                    current = mapping.get(route_key)
                else:
                    current = self.edges.get(current)
            return state

logger = logging.getLogger(__name__)

class QueryComplexity(Enum):
    """Query complexity levels determine routing."""
    SIMPLE = "simple"           # Fact lookup only
    MODERATE = "moderate"       # Graph traversal needed
    COMPLEX = "complex"         # Deep reasoning required

class OrchestratorState(TypedDict, total=False):
    """
    Orchestrator state that flows through the LangGraph.
    Uses TypedDict to ensure type safety across nodes.
    """
    query: str
    query_complexity: QueryComplexity
    retrieved_docs: List[Dict]
    graph_triples: List[Dict]
    neural_proposals: List[Dict]
    verified_facts: List[Dict]
    confidence_score: float
    reasoning_path: List[str]
    synthesis_output: str
    error_message: Optional[str]
    iteration_count: int
    max_iterations: int
    needs_refinement: bool

class LangGraphOrchestrator:
    def __init__(self, retriever, knowledge_graph, symbolic_engine, llm_provider):
        """
        Initialize the LangGraph Orchestrator.
        
        Args:
            retriever: LanceDB retriever instance
            knowledge_graph: Neo4j knowledge graph instance
            symbolic_engine: Symbolic engine instance
            llm_provider: LLM provider (Qwen3 or similar)
        """
        self.retriever = retriever
        self.kg = knowledge_graph
        self.symbolic_engine = symbolic_engine
        self.llm = llm_provider
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.
        
        Nodes:
        - Node A: Query Analysis & Complexity Detection
        - Node B: Document Retrieval (fast path)
        - Node C: Graph Traversal (deep path)
        - Node D: Neural Generation
        - Node E: Symbolic Verification
        - Node F: Synthesis & Report Generation
        - Node G: Refinement Loop (feedback)
        - Node H: Output
        """
        workflow = StateGraph(OrchestratorState)
        
        # Add nodes
        workflow.add_node("analyze_query", self.node_analyze_query)
        workflow.add_node("retrieve_documents", self.node_retrieve_documents)
        workflow.add_node("traverse_graph", self.node_traverse_graph)
        workflow.add_node("neural_generation", self.node_neural_generation)
        workflow.add_node("symbolic_verification", self.node_symbolic_verification)
        workflow.add_node("synthesis", self.node_synthesis)
        workflow.add_node("refinement", self.node_refinement)
        
        # Add edges with conditional routing
        workflow.set_entry_point("analyze_query")
        
        # From analyze_query, route based on complexity
        workflow.add_conditional_edges(
            "analyze_query",
            self.route_by_complexity,
            {
                "fast_path": "retrieve_documents",
                "deep_path": "traverse_graph",
                "end": END
            }
        )
        
        # Fast path: retrieval -> generation -> verification -> synthesis
        workflow.add_edge("retrieve_documents", "neural_generation")
        
        # Deep path: graph traversal -> generation -> verification -> synthesis
        workflow.add_edge("traverse_graph", "neural_generation")
        
        # From neural_generation, verify
        workflow.add_edge("neural_generation", "symbolic_verification")
        
        # From verification, route to synthesis or refinement
        workflow.add_conditional_edges(
            "symbolic_verification",
            self.route_to_synthesis_or_refinement,
            {
                "synthesis": "synthesis",
                "refinement": "refinement"
            }
        )
        
        # From refinement, loop back or end
        workflow.add_conditional_edges(
            "refinement",
            self.route_after_refinement,
            {
                "retrieve_documents": "retrieve_documents",
                "traverse_graph": "traverse_graph",
                "synthesis": "synthesis"
            }
        )
        
        # From synthesis, end
        workflow.add_edge("synthesis", END)
        
        return workflow.compile()
    
    # ==================== NODE DEFINITIONS ====================
    
    def node_analyze_query(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node A: Analyze query and detect complexity.
        Determines routing: Fast Path vs Deep Path.
        """
        query = state.get("query", "")
        logger.info(f"[Node A] Analyzing query: {query[:100]}...")
        
        # Simple heuristics for complexity detection
        keyword_counts = {
            "how_why": len([w for w in query.lower().split() if w in ["how", "why", "what", "when", "examine"]]),
            "complex_terms": len([w for w in query.lower().split() if w in ["mechanism", "pathway", "relationship", "interaction", "correlation"]]),
            "entities": len([w for w in query.lower().split() if w.startswith("-")])
        }
        
        complexity_score = (
            keyword_counts["how_why"] * 0.3 +
            keyword_counts["complex_terms"] * 0.4 +
            keyword_counts["entities"] * 0.3
        )
        
        if complexity_score > 5:
            complexity = QueryComplexity.COMPLEX
        elif complexity_score > 2:
            complexity = QueryComplexity.MODERATE
        else:
            complexity = QueryComplexity.SIMPLE
        
        state["query_complexity"] = complexity
        state["iteration_count"] = 0
        state["max_iterations"] = 3
        
        logger.info(f"[Node A] Query complexity: {complexity.value}")
        
        return state
    
    def route_by_complexity(self, state: OrchestratorState) -> str:
        """
        Routing function: Determine fast vs deep path.
        Fast Path: For simple queries, retrieval-only
        Deep Path: For complex queries, graph + reasoning
        """
        complexity = state.get("query_complexity", QueryComplexity.SIMPLE)
        
        if complexity == QueryComplexity.SIMPLE:
            logger.info("[Routing] Taking FAST PATH (retrieval only)")
            return "fast_path"
        elif complexity == QueryComplexity.MODERATE:
            logger.info("[Routing] Taking DEEP PATH (graph traversal)")
            return "deep_path"
        else:
            logger.info("[Routing] Complex query, analyzing...")
            return "deep_path"
    
    def node_retrieve_documents(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node B: Retrieve relevant documents via hybrid search.
        Fast path for simple information lookup.
        """
        query = state.get("query", "")
        logger.info(f"[Node B] Retrieving documents for: {query[:50]}...")
        
        if not hasattr(self.retriever, 'hybrid_search_papers'):
            logger.error("[Node B] Retriever does not have hybrid_search_papers method")
            state["retrieved_docs"] = []
            state["confidence_score"] = 0.0
            return state
        
        try:
            results = self.retriever.hybrid_search_papers(query, k=10)
        except Exception as e:
            logger.error(f"[Node B] Document retrieval failed: {e}")
            results = []
        
        state["retrieved_docs"] = results
        state["confidence_score"] = 0.7 if results else 0.0
        
        logger.info(f"[Node B] Retrieved {len(results)} documents")
        
        return state
    
    def node_traverse_graph(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node C: Traverse knowledge graph for deep reasoning.
        Finds paths between entities and extracts reasoning pathways.
        """
        query = state.get("query", "")
        logger.info(f"[Node C] Traversing knowledge graph...")
        
        if not hasattr(self.retriever, 'hybrid_search_papers') or not hasattr(self.retriever, 'hybrid_search_triples'):
            logger.error("[Node C] Retriever missing required methods (hybrid_search_papers, hybrid_search_triples)")
            state["retrieved_docs"] = []
            state["graph_triples"] = []
            return state
        
        try:
            docs = self.retriever.hybrid_search_papers(query, k=5)
            state["retrieved_docs"] = docs
        except Exception as e:
            logger.error(f"[Node C] Document retrieval failed: {e}")
            state["retrieved_docs"] = []
        
        try:
            triples = self.retriever.hybrid_search_triples(query, k=20)
            state["graph_triples"] = triples
        except Exception as e:
            logger.error(f"[Node C] Triple retrieval failed: {e}")
            state["graph_triples"] = []
        
        logger.info(f"[Node C] Found {len(state.get('graph_triples', []))} relevant triples")
        
        return state
    
    def node_neural_generation(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node D: Neural Generation - LLM proposes relationships and facts.
        Step 1 of Neural-Symbolic Feedback Loop.
        """
        query = state.get("query", "")
        docs = state.get("retrieved_docs", [])
        triples = state.get("graph_triples", [])
        
        logger.info(f"[Node D] Generating neural proposals...")
        
        # Construct context for LLM
        doc_context = "\n".join([
            f"- {doc.get('title', '')}: {doc.get('abstract', '')[:200]}"
            for doc in docs[:3]
        ])
        
        triple_context = "\n".join([
            f"- {t.get('source')} --{t.get('relation')}-> {t.get('target')}"
            for t in triples[:5]
        ])
        
        # Generate proposals via LLM
        try:
            proposals = self.llm.generate_proposals(
                query=query,
                documents=doc_context,
                triples=triple_context
            )
        except Exception as e:
            logger.error(f"[Node D] Proposal generation failed: {e}")
            proposals = []
        
        # Ensure proposals is a valid type
        if proposals is None:
            proposals = []
        if not isinstance(proposals, list):
            proposals = [proposals] if proposals else []
        
        state["neural_proposals"] = proposals
        
        proposal_count = len(proposals) if isinstance(proposals, list) else 1
        logger.info(f"[Node D] Generated {proposal_count} proposals")
        
        return state
    
    def node_symbolic_verification(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node E: Symbolic Verification - Verify proposals against KG.
        Step 2 of Neural-Symbolic Feedback Loop (The Double-Lock).
        """
        proposals = state.get("neural_proposals", [])
        logger.info(f"[Node E] Verifying {len(proposals)} proposals...")
        
        verified_facts = []
        total_confidence = 0.0
        
        facts = []
        for proposal in proposals:
            if isinstance(proposal, dict) and all(k in proposal for k in ("source", "relation", "target")):
                facts.append(proposal)
            else:
                facts.extend(self._parse_proposal(proposal))
            
        for fact in facts:
            # Step 2: Symbolic Verification
            try:
                status, confidence, details = self.symbolic_engine.verify_neural_proposal(
                    fact,
                    query_context=state.get("query", "")
                )
            except Exception as e:
                logger.warning(f"[Node E] Verification failed for fact {fact}: {e}")
                continue
            
            # Handle both Enum and string status values
            status_value = status.value if hasattr(status, 'value') else str(status)
            if status_value != "contradicted":
                verified_facts.append({
                    "fact": fact,
                    "status": status_value,
                    "confidence": confidence,
                    "details": details
                })
                total_confidence += confidence
        
        state["verified_facts"] = verified_facts
        state["confidence_score"] = (
            total_confidence / len(verified_facts) if verified_facts else 0.0
        )
        state["needs_refinement"] = state["confidence_score"] < 0.6
        
        logger.info(f"[Node E] Verified {len(verified_facts)} facts, confidence: {state['confidence_score']:.2f}")
        
        return state
    
    def node_synthesis(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node F: Synthesis - Generate final report using only verified facts.
        Ensures output is grounded in "Proven Pathways" from knowledge graph.
        """
        query = state.get("query", "")
        verified_facts = state.get("verified_facts", [])
        docs = state.get("retrieved_docs", [])
        
        logger.info(f"[Node F] Synthesizing report from {len(verified_facts)} verified facts...")
        
        # Generate synthesis using only verified facts
        synthesis = self.llm.synthesize_report(
            query=query,
            verified_facts=verified_facts,
            supporting_docs=docs
        )
        
        state["synthesis_output"] = synthesis
        
        logger.info(f"[Node F] Synthesis complete")
        
        return state
    
    def node_refinement(self, state: OrchestratorState) -> OrchestratorState:
        """
        Node G: Refinement Loop - Discovery Refinement (Step 8).
        If verification failed, loop back to find more evidence.
        """
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        logger.info(f"[Node G] Refinement iteration {state['iteration_count']}/{state.get('max_iterations', 3)}")
        
        # Reset state for re-attempt
        state["retrieved_docs"] = []
        state["graph_triples"] = []
        state["verified_facts"] = []
        
        return state
    
    # ==================== ROUTING FUNCTIONS ====================
    
    def route_to_synthesis_or_refinement(self, state: OrchestratorState) -> str:
        """
        After verification: If high confidence, synthesize. Otherwise, refine.
        """
        confidence = state.get("confidence_score", 0.0)
        iteration = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        if confidence > 0.6 or iteration >= max_iterations:
            logger.info(f"[Routing] Proceeding to SYNTHESIS (confidence: {confidence:.2f})")
            return "synthesis"
        else:
            logger.info(f"[Routing] REFINEMENT needed (confidence: {confidence:.2f})")
            return "refinement"
    
    def route_after_refinement(self, state: OrchestratorState) -> str:
        """
        After refinement: Choose next step based on query complexity.
        """
        complexity = state.get("query_complexity", QueryComplexity.SIMPLE)
        
        if complexity == QueryComplexity.SIMPLE:
            return "retrieve_documents"
        else:
            return "traverse_graph"
    
    # ==================== HELPER METHODS ====================
    
    def _parse_proposal(self, proposal_text: str) -> List[Dict]:
        """
        Parse LLM proposal into structured facts.
        Accepts JSON arrays or line-based relation proposals.
        """
        if isinstance(proposal_text, list):
            return proposal_text

        if not isinstance(proposal_text, str):
            return []

        try:
            parsed = json.loads(proposal_text)
            if isinstance(parsed, dict):
                return [parsed]
            if isinstance(parsed, list):
                return [fact for fact in parsed if isinstance(fact, dict)]
        except json.JSONDecodeError as e:
            logger.debug(f"[Parser] JSON parsing failed: {e}. Trying line-based parsing.")
        except Exception as e:
            logger.warning(f"[Parser] Unexpected error during JSON parsing: {e}")
            pass

        facts = []
        for line in proposal_text.splitlines():
            if not line.strip():
                continue
            try:
                if "--" in line and ">" in line:
                    parts = line.strip().split("--")
                    if len(parts) >= 2:
                        source = parts[0].strip()
                        relation_target = parts[1].split(">")
                        if len(relation_target) == 2:
                            relation = relation_target[0].strip().strip("[]-")
                            target = relation_target[1].strip()
                            facts.append({"source": source, "relation": relation, "target": target})
                elif ":" in line and "," in line:
                    content = line.split(":", 1)[1]
                    parts = [p.strip() for p in content.split(",") if p.strip()]
                    if len(parts) == 3:
                        facts.append({"source": parts[0], "relation": parts[1], "target": parts[2]})
            except Exception as e:
                logger.debug(f"[Parser] Line parsing failed for '{line[:50]}...': {e}")
                continue
        return facts
    
    def execute(self, query: str, max_iterations: int = 3) -> Dict:
        """
        Execute the orchestration workflow.
        
        Args:
            query: User query
            max_iterations: Maximum number of refinement iterations (default: 3)
        
        Returns:
            Final state with synthesis output
        """
        initial_state = OrchestratorState(
            query=query,
            query_complexity=QueryComplexity.SIMPLE,
            retrieved_docs=[],
            graph_triples=[],
            verified_facts=[],
            confidence_score=0.0,
            reasoning_path=[],
            synthesis_output="",
            error_message=None,
            iteration_count=0,
            max_iterations=max_iterations,
            needs_refinement=False,
            neural_proposals=[]
        )
        
        try:
            final_state = self.graph.invoke(initial_state)
            return final_state
        except Exception as e:
            logger.exception(f"Orchestration error: {e}")
            initial_state["error_message"] = str(e)
            return initial_state
