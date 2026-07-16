
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.knowledge_essentials import fetch_papers, build_vector_index, search_index
from utils.scientific_intelligence import construct_knowledge_graph
from utils.cross_domain_discovery import CrossDomainDiscoveryEngine
from utils.research_foundation import ResearchIntelligenceFoundation

def main():
    print("Testing pipeline...", flush=True)
    print("Step 1: Fetching papers...", flush=True)
    papers = fetch_papers("customer road to intelligent machines", limit=5)
    print(f"Fetched {len(papers)} papers!", flush=True)
    for i, p in enumerate(papers):
        print(f"Paper {i+1}: {p['title']}", flush=True)

    print("\nStep 2: Building vector index...", flush=True)
    index, embeddings = build_vector_index(papers)
    print("Index built!", flush=True)

    print("\nStep3: Constructing knowledge graph...", flush=True)
    kg = construct_knowledge_graph(papers)
    print(f"Graph has {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges!", flush=True)

    print("\nStep4: Running discovery engine...", flush=True)
    engine = CrossDomainDiscoveryEngine(kg, papers)
    discoveries = engine.get_all_discoveries()
    print(f"Found {len(discoveries)} discoveries!", flush=True)
    if discoveries:
        print(f"First discovery: {discoveries[0]}", flush=True)

    print("\nStep5: Generating reports...", flush=True)
    foundation = ResearchIntelligenceFoundation()
    baseline = foundation.generate_baseline_report("customer road to intelligent machines", papers)
    print("Generated baseline report!", flush=True)
    print(f"Baseline report length: {len(baseline)}", flush=True)

    if discoveries:
        ns_report = foundation.run_agentic_loop("customer road to intelligent machines", discoveries[0], papers)
        print("Generated neuro-symbolic report!", flush=True)
        print(f"Neuro-symbolic report length: {len(ns_report)}", flush=True)

    print("\nPipeline test completed!", flush=True)


if __name__ == "__main__":
    main()
