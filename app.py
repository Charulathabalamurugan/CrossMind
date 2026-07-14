import streamlit as st
import networkx as nx
import json
import os
import time


def notify(message, icon=None, status='info'):
    """Display a toast if available, otherwise use Streamlit info/error messaging."""
    if hasattr(st, 'toast'):
        try:
            if icon:
                st.toast(message, icon=icon)
            else:
                st.toast(message)
            return
        except Exception:
            pass

    if status == 'error':
        st.error(message)
    elif status == 'success':
        st.success(message)
    else:
        st.info(message)


# Set up page configurations
st.set_page_config(
    page_title="CrossMind - Neuro-Symbolic Scientific Discovery",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our custom modules
from utils.knowledge_essentials import fetch_papers, build_vector_index, search_index
from utils.scientific_intelligence import construct_knowledge_graph
from utils.cross_domain_discovery import CrossDomainDiscoveryEngine
from utils.research_foundation import ResearchIntelligenceFoundation
from utils.lancedb_retrieval import LanceDBRetriever
from utils.neo4j_graph import Neo4jKnowledgeGraph
from utils.symbolic_engine import SymbolicEngine
from utils.qwen3_provider import Qwen3Provider
from utils.langgraph_orchestrator import LangGraphOrchestrator
from utils.visualization import networkx_to_pyvis_html, path_metrics_figure
from utils.source_retrieval import fetch_multi_source_papers
from utils.enterprise_ops import ObservabilityRecorder, GovernanceEngine

CACHE_TTL_SECONDS = 3600

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def cached_fetch_papers(query, limit, api_key=None):
    return fetch_papers(query, limit)

@st.cache_data(ttl=CACHE_TTL_SECONDS)
def cached_build_graph(papers, api_key=None):
    return construct_knowledge_graph(papers, api_key=api_key)


def extract_graph_triples(graph):
    triples = []
    for u, v, data in graph.edges(data=True):
        triples.append({
            "source": u,
            "relation": data.get("type", "ASSOCIATED_WITH"),
            "target": v,
            "confidence": 0.75,
            "source_papers": data.get("paper_ids", [])
        })
    return triples


def extract_entities_for_neo4j(graph):
    entities = []
    for node, data in graph.nodes(data=True):
        entities.append({
            "name": node,
            "label": data.get("label", "Concept"),
            "description": data.get("description", ""),
            "source_papers": data.get("paper_ids", [])
        })
    return entities


def build_orchestrator(retriever, knowledge_graph, symbolic_engine, qwen_provider):
    return LangGraphOrchestrator(
        retriever=retriever,
        knowledge_graph=knowledge_graph,
        symbolic_engine=symbolic_engine,
        llm_provider=qwen_provider
    )

# Apply Premium Custom CSS
st.markdown("""
<style>
    /* Main Layout */
    .stApp {
        background-color: #0a0b10;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0f111a !important;
        border-right: 1px solid #1e293b;
    }
    
    /* Header Card */
    .main-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 50%, #0f172a 100%);
        border: 1px solid #3730a3;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px rgba(79, 70, 229, 0.15);
    }
    .main-title {
        font-size: 2.2rem !important;
        font-weight: 800 !important;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .main-subtitle {
        color: #94a3b8;
        font-size: 1rem;
    }
    
    /* Pipeline Step Tracker */
    .step-card {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 12px 16px;
        margin-bottom: 8px;
        transition: all 0.3s ease;
    }
    .step-card.active {
        border-color: #6366f1;
        background-color: #1e1b4b;
        box-shadow: 0 0 10px rgba(99, 102, 241, 0.2);
    }
    .step-card.success {
        border-color: #10b981;
        background-color: #064e3b;
    }
    .step-title {
        font-weight: 600;
        font-size: 0.9rem;
    }
    .step-desc {
        font-size: 0.75rem;
        color: #94a3b8;
    }
    
    /* Metric Cards */
    .metric-container {
        background-color: #0f111a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #a78bfa;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        border-bottom: 1px solid #1e293b;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-weight: 600 !important;
        font-size: 0.95rem;
        color: #94a3b8;
        background-color: transparent !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #e2e8f0;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #818cf8;
        border-bottom: 2px solid #818cf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Banner
st.markdown("""
<div class="main-header">
    <div class="main-title">🧠 CrossMind</div>
    <div class="main-subtitle">Neuro-Symbolic Scientific Discovery Engine & Cross-Domain Hypothesis Generator</div>
</div>
""", unsafe_allow_html=True)

# Session States
if "papers" not in st.session_state:
    st.session_state.papers = None
if "graph" not in st.session_state:
    st.session_state.graph = None
if "discoveries" not in st.session_state:
    st.session_state.discoveries = []
if "pipeline_stage" not in st.session_state:
    st.session_state.pipeline_stage = "idle"
if "selected_path_idx" not in st.session_state:
    st.session_state.selected_path_idx = 0
if "baseline_report" not in st.session_state:
    st.session_state.baseline_report = ""
if "ns_report" not in st.session_state:
    st.session_state.ns_report = ""
if "advanced_report" not in st.session_state:
    st.session_state.advanced_report = ""
if "orchestrator_state" not in st.session_state:
    st.session_state.orchestrator_state = {}
if "observability_summary" not in st.session_state:
    st.session_state.observability_summary = ""
if "review_required" not in st.session_state:
    st.session_state.review_required = False
if "governance_result" not in st.session_state:
    st.session_state.governance_result = {}

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Engine Settings")

# API Keys
gemini_key = st.sidebar.text_input("Gemini API Key", type="password", help="Needed for live relation extraction and agentic hypothesis generation.")
qwen_key = st.sidebar.text_input("Qwen3 API Key", type="password", help="Needed for advanced Qwen3 reasoning and synthesis.")
s2_key = st.sidebar.text_input("Semantic Scholar Key (Optional)", type="password", help="Enables higher API rate limits.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔐 Governance")
st.sidebar.checkbox("Enable review workflow", value=True, key="enable_review")

# Neo4j Settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧠 Neo4j Knowledge Graph")
neo4j_uri = st.sidebar.text_input("Neo4j URI", value="bolt://localhost:7687")
neo4j_user = st.sidebar.text_input("Neo4j Username", value="neo4j")
neo4j_password = st.sidebar.text_input("Neo4j Password", type="password")
neo4j_db = st.sidebar.text_input("Neo4j Database", value="neo4j")

# Query configuration
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Research Query")

# Presets
preset_options = [
    "analyze customer road to intelligent machines",
    "graphene pacemakers biocompatible energy harvesting",
    "Custom Query"
]
selected_preset = st.sidebar.selectbox("Choose a Discovery Scenario", preset_options)

if selected_preset == "Custom Query":
    search_query = st.sidebar.text_input("Enter your Custom Query", "biodegradable plastics packaging marine microbes")
else:
    search_query = selected_preset

top_k_papers = st.sidebar.slider("Papers to Ingest", min_value=5, max_value=25, value=10, step=5)

run_button = st.sidebar.button("🧠 Initiate Discovery Pipeline", use_container_width=True)

# Layout: Main columns
col_run, col_status = st.columns([3, 1])

# Left column: Content Tabs
with col_run:
    tab_graph, tab_report, tab_papers = st.tabs(["🌐 Knowledge Graph Discovery", "📄 Scientific Hypothesis & Reports", "📚 Literature database"])

# Right column: Pipeline Live Tracker
with col_status:
    st.markdown("### 🔄 Pipeline Progress")
    
    stages = [
        ("essentials", "1. Knowledge Essentials", "Retrieving papers & embedding via BGE-M3..."),
        ("intelligence", "2. Scientific Intelligence", "SciBERT & NetworkX extraction..."),
        ("discovery", "3. Cross-Domain Discovery", "Evaluating symbolic Cypher rules..."),
        ("foundation", "4. Research Foundation", "Running Agentic AI Synthesis loop...")
    ]
    
    stage_placeholders = {}
    for stage_id, name, desc in stages:
        stage_placeholders[stage_id] = st.empty()
        
    def update_tracker(current_stage, status="pending"):
        for sid, name, desc in stages:
            if sid == current_stage:
                if status == "running":
                    class_name = "active"
                    icon = "⏳"
                elif status == "success":
                    class_name = "success"
                    icon = "✅"
                else:
                    class_name = ""
                    icon = "⏳"
                stage_placeholders[sid].markdown(f"""
                <div class="step-card {class_name}">
                    <div class="step-title">{icon} {name}</div>
                    <div class="step-desc">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # check if prior
                is_prior = stages.index((sid, name, desc)) < stages.index([s for s in stages if s[0] == current_stage][0])
                if is_prior:
                    stage_placeholders[sid].markdown(f"""
                    <div class="step-card success">
                        <div class="step-title">✅ {name}</div>
                        <div class="step-desc">Completed</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    stage_placeholders[sid].markdown(f"""
                    <div class="step-card">
                        <div class="step-title">💤 {name}</div>
                        <div class="step-desc">{desc}</div>
                    </div>
                    """, unsafe_allow_html=True)

    # Initial render
    update_tracker("essentials", "pending")

# Execution trigger
if run_button:
    if gemini_key:
        os.environ["GEMINI_API_KEY"] = gemini_key
    if s2_key:
        os.environ["SEMANTIC_SCHOLAR_API_KEY"] = s2_key

    current_stage = "essentials"
    try:
        notify("Connecting to Scientific APIs...", icon="🌐")
        update_tracker(current_stage, "running")

        recorder = ObservabilityRecorder()
        governance = GovernanceEngine()
        try:
            papers = fetch_multi_source_papers(search_query, limit=top_k_papers)
        except Exception:
            papers = cached_fetch_papers(search_query, top_k_papers, api_key=gemini_key)
        st.session_state.papers = papers
        if not papers:
            notify("No papers retrieved. Using fallback data and offline mode.", status="error")

        recorder.record("retrieval_completed", {"query": search_query, "paper_count": len(papers)})

        notify("Generating BGE-M3 Embeddings...", icon="🧠")
        index, embeddings = build_vector_index(papers)
        if index is not None:
            papers = search_index(search_query, index, embeddings, papers, top_k=top_k_papers)
            st.session_state.papers = papers

        update_tracker(current_stage, "success")
        notify(f"Retrieved {len(papers)} papers using BGE-M3 & FAISS Index!", icon="✅", status="success")
        time.sleep(0.5)

        current_stage = "intelligence"
        notify("Running SciBERT NER & Building NetworkX graph...", icon="🧬")
        update_tracker(current_stage, "running")

        G = cached_build_graph(papers, api_key=gemini_key if gemini_key else None)
        st.session_state.graph = G

        update_tracker(current_stage, "success")
        notify(f"Knowledge Graph populated with {G.number_of_nodes()} concepts!", icon="✅", status="success")
        time.sleep(0.5)

        current_stage = "discovery"
        notify("Evaluating Symbolic Rules for bridging concepts...", icon="🧩")
        update_tracker(current_stage, "running")

        engine = CrossDomainDiscoveryEngine(G, papers)
        discoveries = engine.get_all_discoveries()
        st.session_state.discoveries = discoveries
        st.session_state.selected_path_idx = 0

        update_tracker(current_stage, "success")
        notify(f"Discovered {len(discoveries)} hidden pathways!", icon="✅", status="success")
        time.sleep(0.5)

        current_stage = "foundation"
        notify("Starting Agentic AI Generator-Critic Loop...", icon="🤖")
        update_tracker(current_stage, "running")

        foundation = ResearchIntelligenceFoundation(api_key=gemini_key if gemini_key else None)
        st.session_state.baseline_report = foundation.generate_baseline_report(search_query, papers)

        lance_retriever = LanceDBRetriever(db_path="./lancedb_store")
        lance_retriever.add_papers(papers)
        lance_retriever.add_knowledge_triples(extract_graph_triples(G))

        neo_kg = Neo4jKnowledgeGraph(
            uri=neo4j_uri,
            username=neo4j_user,
            password=neo4j_password or "password",
            database=neo4j_db
        )
        if neo_kg.driver:
            neo_kg.incremental_update(extract_entities_for_neo4j(G), extract_graph_triples(G))
        else:
            neo_kg.incremental_update(extract_entities_for_neo4j(G), extract_graph_triples(G))

        symbolic_engine = SymbolicEngine(knowledge_graph=neo_kg)
        qwen_provider = Qwen3Provider(api_key=qwen_key or os.environ.get("QWEN_API_KEY"))

        orchestrator = build_orchestrator(
            retriever=lance_retriever,
            knowledge_graph=neo_kg,
            symbolic_engine=symbolic_engine,
            qwen_provider=qwen_provider
        )

        try:
            orchestrator_state = orchestrator.execute(search_query)
        except Exception as orch_err:
            orchestrator_state = {
                "error_message": str(orch_err),
                "synthesis_output": "",
                "confidence_score": 0.0,
                "iteration_count": 0,
                "verified_facts": [],
            }
        st.session_state.orchestrator_state = orchestrator_state
        governance_result = governance.evaluate_confidence(
            orchestrator_state.get("confidence_score", 0.0),
            evidence_count=len(orchestrator_state.get("verified_facts", [])),
            sources=[p.get("source") for p in papers if p.get("source")]
        )
        st.session_state.governance_result = governance_result
        st.session_state.review_required = bool(governance_result.get("review_required") and st.session_state.get("enable_review", True))
        st.session_state.observability_summary = recorder.summary()
        recorder.record("governance_evaluated", governance_result)

        if discoveries:
            selected_path = discoveries[0]
            st.session_state.ns_report = foundation.run_agentic_loop(search_query, selected_path, papers)
        else:
            st.session_state.ns_report = foundation.run_agentic_loop(
                search_query,
                {"path": ["query_subject"], "labels": ["Subject"]},
                papers
            )

        if orchestrator_state.get("synthesis_output"):
            st.session_state.advanced_report = orchestrator_state.get("synthesis_output")
        else:
            st.session_state.advanced_report = st.session_state.ns_report

        update_tracker(current_stage, "success")
        notify("Research Synthesis Completed!", icon="🎉", status="success")
        st.session_state.pipeline_stage = "done"
        st.rerun()
    except Exception as e:
        st.error(f"Pipeline failed during '{current_stage}' stage: {str(e)}")
        notify("The discovery pipeline encountered an error. Check console logs and API connectivity.", status="error")
        st.session_state.pipeline_stage = "error"
        st.stop()

# RENDER DASHBOARD TABS
# Tab 1: Graph View
with tab_graph:
    if st.session_state.graph is not None:
        G = st.session_state.graph
        discoveries = st.session_state.discoveries
        
        # Render Graph statistics
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{G.number_of_nodes()}</div>
                <div class="metric-label">Concepts (Nodes)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_g2:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{G.number_of_edges()}</div>
                <div class="metric-label">Relations (Edges)</div>
            </div>
            """, unsafe_allow_html=True)
        with col_g3:
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-value">{len(discoveries)}</div>
                <div class="metric-label">Discovered Pathways</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")
        
        # Left Panel: Pathway Explorer | Right Panel: Vis.js Interactive Graph
        col_explorer, col_network = st.columns([1, 2])
        
        with col_explorer:
            st.markdown("### 🧩 Discovered Connections")
            if discoveries:
                path_titles = []
                for idx, disc in enumerate(discoveries):
                    path_str = " ➔ ".join(disc["path"])
                    score = disc["metrics"]["score"]
                    path_titles.append(f"Bridge {idx+1}: {disc['path'][0]} ➔ {disc['path'][-1]} (Score: {score:.2f})")
                    
                selected_idx = st.selectbox(
                    "Select Pathway to Inspect & Highlight",
                    range(len(discoveries)),
                    format_func=lambda idx: path_titles[idx]
                )
                st.session_state.selected_path_idx = selected_idx
                
                # Show path details
                path_data = discoveries[selected_idx]
                st.markdown(f"**Selected Path:** `{ ' ➔ '.join(path_data['path']) }`")
                
                # Render Path node-by-node details
                st.markdown("##### 📍 Path Details")
                for i in range(len(path_data["path"]) - 1):
                    u = path_data["path"][i]
                    v = path_data["path"][i+1]
                    rel = path_data["relations"][i]
                    u_label = path_data["labels"][i]
                    v_label = path_data["labels"][i+1]
                    st.markdown(f"- **{u}** ({u_label})  \n  `-[{rel}]->`  \n  **{v}** ({v_label})")
                    
                # Render metrics
                st.markdown("##### 📊 Scores")
                m = path_data["metrics"]
                st.markdown(f"""
                - **Composite Discovery Score**: `{m['score']:.3f}`
                - **Semantic Similarity**: `{m['similarity']:.3f}`
                - **Citation Weight**: `{m['citation_weight']:.3f}`
                - **Path Strength**: `{m['path_strength']:.3f}`
                - **Supporting Citations**: `{m['total_citations']}`
                """)
                try:
                    fig = path_metrics_figure(path_data)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass
            else:
                st.warning("No multi-hop interdisciplinary pathways discovered. Try increasing the number of papers to ingest or run a different query.")
                
        with col_network:
            st.markdown("### 🌐 Interactive Concept Map")
            
            # Serialize NetworkX graph to JSON for Vis.js
            nodes = []
            edges = []
            
            # Identify current path to highlight
            highlighted_nodes = set()
            highlighted_edges = set()
            
            if discoveries and len(discoveries) > st.session_state.selected_path_idx:
                p_data = discoveries[st.session_state.selected_path_idx]
                path = p_data["path"]
                for n in path:
                    highlighted_nodes.add(n)
                for i in range(len(path) - 1):
                    highlighted_edges.add((path[i], path[i+1]))
                    highlighted_edges.add((path[i+1], path[i])) # handle undirected mapping
            
            # Set Node Colors based on Category
            color_palette = {
                "CustomerNeed": "#f43f5e",   # Rose
                "Obstacle": "#f59e0b",       # Amber
                "Mechanism": "#10b981",      # Emerald
                "Technology": "#3b82f6",     # Blue
                "Material": "#a855f7",       # Purple
                "Property": "#06b6d4",       # Cyan
                "Device": "#ec4899"          # Pink
            }
            
            for node, data in G.nodes(data=True):
                label = data.get("label", "Concept")
                node_color = color_palette.get(label, "#94a3b8")
                
                # Check if highlighted
                is_hi = node in highlighted_nodes
                border_color = "#ffffff" if is_hi else node_color
                size = 35 if is_hi else 20
                shadow_options = {"enabled": True, "color": "rgba(255,255,255,0.4)"} if is_hi else {"enabled": True}
                
                nodes.append({
                    "id": node,
                    "label": node,
                    "title": f"<b>{node.upper()}</b> ({label})<br>{data.get('description', '')}<br>Appeared in {len(data.get('paper_ids', []))} paper(s)",
                    "color": {
                        "background": node_color,
                        "border": border_color,
                        "highlight": {"background": "#ff007f", "border": "#ffffff"}
                    },
                    "size": size,
                    "borderWidth": 3 if is_hi else 1,
                    "shadow": shadow_options
                })
                
            for u, v, data in G.edges(data=True):
                rel_type = data.get("type", "ASSOCIATED_WITH")
                
                # Check if highlighted
                is_edge_hi = (u, v) in highlighted_edges or (v, u) in highlighted_edges
                edge_color = "#ff007f" if is_edge_hi else "#475569"
                edge_width = 4 if is_edge_hi else 1.5
                
                edges.append({
                    "from": u,
                    "to": v,
                    "label": rel_type,
                    "color": {
                        "color": edge_color,
                        "highlight": "#ff007f",
                        "hover": "#ff007f"
                    },
                    "width": edge_width,
                    "arrows": {"to": {"enabled": True, "scaleFactor": 0.4}}
                })
                
            # Embed Vis.js network graph using iframe
            nodes_json = json.dumps(nodes)
            edges_json = json.dumps(edges)
            
            graph_html = networkx_to_pyvis_html(G, highlighted_nodes=highlighted_nodes, highlighted_edges=highlighted_edges, height=600)
            st.components.v1.html(graph_html, height=600, scrolling=False)
    else:
        st.info("Initiate the Discovery Pipeline in the sidebar to populate the Knowledge Graph.")

# Tab 2: Scientific Reports & Hypothesis Comparison
with tab_report:
    if st.session_state.baseline_report or st.session_state.ns_report:
        col_baseline, col_ns = st.columns(2)
        
        with col_baseline:
            st.markdown("### 📄 GraphRAG Baseline Synthesis")
            st.markdown("""
            *This is the standard vector-matching report baseline. It treats paper abstracts as a flat list,
            relying solely on dense similarity to summarize the topics without structural rule reasoning.*
            """)
            st.markdown("---")
            st.markdown(st.session_state.baseline_report)
            
        with col_ns:
            st.markdown("### 🚀 Neuro-Symbolic Hypothesis & Report")
            st.markdown("""
            *This report combines graph-based discovery with Qwen3-style reasoning and symbolic verification.
            It is grounded in the extracted pathway and filtered through the neural-symbolic feedback loop.*
            """)
            st.markdown("---")
            report_to_display = st.session_state.get("advanced_report") or st.session_state.ns_report
            st.markdown(report_to_display)

            orchestrator_state = st.session_state.get("orchestrator_state", {})
            if orchestrator_state:
                st.markdown("### 🔎 Orchestrator Diagnostics")
                st.markdown(f"- **Final Confidence Score:** `{orchestrator_state.get('confidence_score', 0.0):.2f}`")
                st.markdown(f"- **Iteration Count:** `{orchestrator_state.get('iteration_count', 0)}`")
                st.markdown(f"- **Selected Pathway Refined:** `{len(orchestrator_state.get('verified_facts', []))} verified facts`")
                if orchestrator_state.get("error_message"):
                    st.error(f"Orchestrator error: {orchestrator_state.get('error_message')}")

                governance_result = st.session_state.get("governance_result", {})
                if governance_result:
                    st.markdown("### 🛡️ Governance Review")
                    st.markdown(f"- **Policy:** `{governance_result.get('policy', 'research')}`")
                    st.markdown(f"- **Risk Level:** `{governance_result.get('risk_level', 'unknown')}`")
                    st.markdown(f"- **Review Required:** `{governance_result.get('review_required', False)}`")
                    if governance_result.get("review_required"):
                        st.warning("Low-confidence or low-evidence results were routed to human review.")

                if st.session_state.get("observability_summary"):
                    st.markdown("### 📈 Observability Trace")
                    st.markdown(st.session_state.observability_summary)
    else:
        st.info("Initiate the Discovery Pipeline in the sidebar to generate Scientific Reports.")

# Tab 3: Literature Database
with tab_papers:
    if st.session_state.papers is not None:
        st.markdown("### 📚 Ingested Publication Repository")
        
        # Display data table
        papers_display = []
        for idx, p in enumerate(st.session_state.papers):
            papers_display.append({
                "Rank": idx + 1,
                "Title": p["title"],
                "Authors": p["authors"],
                "Year": p["year"],
                "Citations": p["citation_count"],
                "Venue": p["venue"],
                "Semantic Similarity": f"{p.get('similarity_score', 0):.3f}"
            })
            
        st.dataframe(papers_display, use_container_width=True)
    else:
        st.info("No papers ingested yet. Use the sidebar controller to search.")
