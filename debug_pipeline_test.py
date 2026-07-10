from utils.knowledge_essentials import fetch_papers, build_vector_index, search_index
from utils.scientific_intelligence import construct_knowledge_graph
from utils.cross_domain_discovery import CrossDomainDiscoveryEngine
from utils.research_foundation import ResearchIntelligenceFoundation
from utils.lancedb_retrieval import LanceDBRetriever
from utils.neo4j_graph import Neo4jKnowledgeGraph
from utils.symbolic_engine import SymbolicEngine
from utils.qwen3_provider import Qwen3Provider
from utils.langgraph_orchestrator import LangGraphOrchestrator

query = 'biodegradable plastics packaging marine microbes'
print('Fetching papers...')
papers = fetch_papers(query, limit=5)
print('Got', len(papers), 'papers')
index, embeddings = build_vector_index(papers)
print('Index built:', index is not None)
papers = search_index(query, index, embeddings, papers, top_k=5)
print('Search yielded', len(papers), 'papers')
G = construct_knowledge_graph(papers)
print('Graph nodes', G.number_of_nodes(), 'edges', G.number_of_edges())

engine = CrossDomainDiscoveryEngine(G, papers)
discoveries = engine.get_all_discoveries()
print('Discoveries', len(discoveries))
if discoveries:
    print('Sample pathway:', discoveries[0]['path'])

baseline_report = ResearchIntelligenceFoundation(api_key=None).generate_baseline_report(query, papers)
print('Baseline report length', len(baseline_report))

retriever = LanceDBRetriever(db_path='./tmp_lancedb_store')
retriever.add_papers(papers)
retriever.add_knowledge_triples([{'source':'a','relation':'RELATES','target':'b','confidence':0.5,'source_papers':['x']}])
print('Retriever stats', retriever.get_statistics())

neo = Neo4jKnowledgeGraph(uri='bolt://localhost:7687', username='neo4j', password='password', database='neo4j')
print('Neo4j connected', neo.is_connected)

sym = SymbolicEngine(neo)
qwen = Qwen3Provider(api_key=None)
orchestrator = LangGraphOrchestrator(retriever=retriever, knowledge_graph=neo, symbolic_engine=sym, llm_provider=qwen)
final_state = orchestrator.execute(query)
print('Orchestrator confidence', final_state.get('confidence_score'))
print('Orchestrator synthesis output length', len(final_state.get('synthesis_output','')))
print('Orchestrator verified facts', len(final_state.get('verified_facts', [])))
print('Done')
