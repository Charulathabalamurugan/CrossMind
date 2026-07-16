
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.knowledge_essentials import get_synthetic_papers, _synthetic_customer_machine_papers
from utils.scientific_intelligence import construct_knowledge_graph
from utils.cross_domain_discovery import CrossDomainDiscoveryEngine
from utils.research_foundation import ResearchIntelligenceFoundation

def main():
    print("Testing pipeline with synthetic papers...", flush=True)
    
    print("Step 1: Getting synthetic papers...", flush=True)
    papers = _synthetic_customer_machine_papers()
    print(f"Fetched {len(papers)} papers!", flush=True)
    for i, p in enumerate(papers):
        print(f"Paper {i+1}: {p['title']}", flush=True)

    print("\nStep 3: Constructing knowledge graph...", flush=True)
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
    print("--- Baseline Report Snippet ---", flush=True)
    print(baseline[:500].encode('ascii', errors='replace').decode('ascii'), flush=True)

    if discoveries:
        ns_report = foundation.run_agentic_loop("customer road to intelligent machines", discoveries[0], papers)
        print("Generated neuro-symbolic report!", flush=True)
        print(f"Neuro-symbolic report length: {len(ns_report)}", flush=True)
        print("--- Neuro-Symbolic Report Snippet ---", flush=True)
        print(ns_report[:500].encode('ascii', errors='replace').decode('ascii'), flush=True)

    print("\nPipeline test completed!", flush=True)


if __name__ == "__main__":
    main()
