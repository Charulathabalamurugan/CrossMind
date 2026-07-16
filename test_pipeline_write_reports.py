
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from utils.knowledge_essentials import _synthetic_customer_machine_papers
from utils.scientific_intelligence import construct_knowledge_graph
from utils.cross_domain_discovery import CrossDomainDiscoveryEngine
from utils.research_foundation import ResearchIntelligenceFoundation

def main():
    print("Testing pipeline with synthetic papers and writing reports to files...", flush=True)
    
    print("Step 1: Getting synthetic papers...", flush=True)
    papers = _synthetic_customer_machine_papers()
    print(f"Fetched {len(papers)} papers!", flush=True)

    print("\nStep 3: Constructing knowledge graph...", flush=True)
    kg = construct_knowledge_graph(papers)
    print(f"Graph has {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges!", flush=True)

    print("\nStep4: Running discovery engine...", flush=True)
    engine = CrossDomainDiscoveryEngine(kg, papers)
    discoveries = engine.get_all_discoveries()
    print(f"Found {len(discoveries)} discoveries!", flush=True)

    print("\nStep5: Generating reports...", flush=True)
    foundation = ResearchIntelligenceFoundation()
    baseline = foundation.generate_baseline_report("customer road to intelligent machines", papers)

    if discoveries:
        ns_report = foundation.run_agentic_loop("customer road to intelligent machines", discoveries[0], papers)
    else:
        ns_report = "No discoveries found."

    # Write reports to files
    with open("baseline_report.txt", "w", encoding="utf-8") as f:
        f.write(baseline)
    print("Wrote baseline_report.txt", flush=True)
    
    with open("neuro_symbolic_report.txt", "w", encoding="utf-8") as f:
        f.write(ns_report)
    print("Wrote neuro_symbolic_report.txt", flush=True)
    
    print("\nPipeline test completed! Check the output files!", flush=True)


if __name__ == "__main__":
    main()
