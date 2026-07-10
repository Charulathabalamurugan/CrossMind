import networkx as nx
import numpy as np
import math

class CrossDomainDiscoveryEngine:
    def __init__(self, graph, papers_dict):
        """
        Initializes the Discovery Engine with a NetworkX graph and a paper metadata dictionary
        to evaluate citation and similarity statistics.
        """
        self.G = graph
        self.papers = {p["paper_id"]: p for p in papers_dict}

    def get_node_papers_citations(self, node_name):
        """
        Calculates the aggregate citation count for a node based on the papers it appears in.
        """
        if not self.G.has_node(node_name):
            return 0
        paper_ids = self.G.nodes[node_name].get("paper_ids", [])
        citations = 0
        for pid in paper_ids:
            if pid in self.papers:
                citations += self.papers[pid].get("citation_count", 0)
        return citations

    def calculate_path_score(self, path, edge_types):
        """
        Computes the composite score for a reasoning path.
        Formula: Score = 0.4 * Similarity + 0.3 * CitationWeight + 0.3 * PathStrength
        """
        # 1. Calculate Average Semantic Similarity of supporting papers
        similarities = []
        pids_in_path = set()
        
        # Collect all paper IDs along the path
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            # Find edge data
            edge_data = self.G.get_edge_data(u, v)
            if edge_data:
                # networkx MultiDiGraph returns a dict of keys, we inspect the first one
                first_key = list(edge_data.keys())[0]
                pids = edge_data[first_key].get("paper_ids", [])
                for pid in pids:
                    pids_in_path.add(pid)
                    if pid in self.papers and "similarity_score" in self.papers[pid]:
                        similarities.append(self.papers[pid]["similarity_score"])
                        
        avg_similarity = np.mean(similarities) if similarities else 0.5

        # 2. Calculate Citation Weight (Log-normalized aggregate citation counts)
        total_citations = 0
        for pid in pids_in_path:
            if pid in self.papers:
                total_citations += self.papers[pid].get("citation_count", 0)
                
        # Log-scale normalization: maps [0, 1000+] to [0.1, 1.0]
        citation_weight = min(1.0, math.log(1 + total_citations) / 8.0) if total_citations > 0 else 0.1

        # 3. Path Strength (shorter paths are stronger, penalized if nodes are highly cluttered)
        # Length penalty: length 2 (3 nodes) -> 1.0, length 3 (4 nodes) -> 0.8
        path_len = len(path) - 1
        length_factor = 1.0 if path_len == 2 else 0.8
        
        # Node clustering/degree penalty to prevent hub-nodes from dominating
        degrees = [self.G.degree(node) for node in path]
        avg_degree = np.mean(degrees)
        degree_penalty = max(0.5, 1.0 - (avg_degree / 30.0)) # penalize hubs with degree > 15
        
        path_strength = length_factor * degree_penalty

        # Composite score
        composite_score = (0.4 * avg_similarity) + (0.3 * citation_weight) + (0.3 * path_strength)
        
        return {
            "score": float(composite_score),
            "similarity": float(avg_similarity),
            "citation_weight": float(citation_weight),
            "path_strength": float(path_strength),
            "total_citations": total_citations
        }

    def discover_customer_to_machine_bridges(self):
        """
        Executes symbolic path reasoning for industrial automation.
        Rule: MATCH (c:CustomerNeed)-[:FACES_CHALLENGE]->(o:Obstacle)-[:RESOLVED_BY]->(m:Mechanism)<-[:REQUIRES]-(t:Technology)
        """
        discovered_paths = []
        
        # Find all customer need nodes
        customer_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("label") == "CustomerNeed"]
        tech_nodes = [n for n, attr in self.G.nodes(data=True) if attr.get("label") == "Technology"]
        
        for c in customer_nodes:
            for t in tech_nodes:
                # Check if they are already directly linked
                if self.G.has_edge(c, t) or self.G.has_edge(t, c):
                    continue # Skip direct links - we want hidden interdisciplinary discoveries
                    
                # Search for 3-hop bridging paths: c -> o -> m <- t
                # Since edge directions vary, we check undirected pathways for general bridging
                undirected_G = self.G.to_undirected()
                try:
                    paths = list(nx.all_simple_paths(undirected_G, source=c, target=t, cutoff=3))
                    for path in paths:
                        if len(path) == 4: # 3 hops (c, o, m, t)
                            o, m = path[1], path[2]
                            
                            # Verify labels: o must be Obstacle/Property, m must be Mechanism/Property
                            o_label = self.G.nodes[o].get("label")
                            m_label = self.G.nodes[m].get("label")
                            
                            if o_label in ["Obstacle", "Property"] and m_label in ["Mechanism", "Property"]:
                                edge_types = []
                                # Collect relation names
                                for idx in range(3):
                                    u, v = path[idx], path[idx+1]
                                    e_data = self.G.get_edge_data(u, v) or self.G.get_edge_data(v, u)
                                    if e_data:
                                        first_key = list(e_data.keys())[0]
                                        edge_types.append(e_data[first_key].get("type", "ASSOCIATED_WITH"))
                                    else:
                                        edge_types.append("ASSOCIATED_WITH")
                                        
                                score_details = self.calculate_path_score(path, edge_types)
                                
                                discovered_paths.append({
                                    "path": path,
                                    "labels": ["CustomerNeed", o_label, m_label, "Technology"],
                                    "relations": edge_types,
                                    "metrics": score_details
                                })
                except nx.NetworkXNoPath:
                    continue
                    
        # Sort by score descending
        discovered_paths.sort(key=lambda x: x["metrics"]["score"], reverse=True)
        return discovered_paths

    def discover_material_to_device_bridges(self):
        """
        Executes symbolic path reasoning for material devices.
        Rule: MATCH (m:Material)-[:HAS_PROPERTY]->(p:Property)-[:USED_IN]->(d:Device)-[:ASSOCIATED_WITH]->(i:Device)
        """
        discovered_paths = []
        
        materials = [n for n, attr in self.G.nodes(data=True) if attr.get("label") == "Material"]
        devices = [n for n, attr in self.G.nodes(data=True) if attr.get("label") == "Device"]
        
        for m in materials:
            for d in devices:
                # We look for a device d which is implantable or active (non-trivial)
                d_desc = self.G.nodes[d].get("description", "").lower()
                d_name = d.lower()
                if not any(
                    term in d_desc or term in d_name
                    for term in ("implant", "wearable", "active", "packaging", "monitor", "biosensor", "sensor")
                ):
                    continue
                    
                if self.G.has_edge(m, d) or self.G.has_edge(d, m):
                    continue
                    
                undirected_G = self.G.to_undirected()
                try:
                    paths = list(nx.all_simple_paths(undirected_G, source=m, target=d, cutoff=3))
                    for path in paths:
                        if len(path) == 4:
                            prop, dev_mid = path[1], path[2]
                            
                            prop_label = self.G.nodes[prop].get("label")
                            dev_mid_label = self.G.nodes[dev_mid].get("label")
                            
                            if prop_label == "Property" and dev_mid_label in ["Device", "Mechanism"]:
                                edge_types = []
                                for idx in range(3):
                                    u, v = path[idx], path[idx+1]
                                    e_data = self.G.get_edge_data(u, v) or self.G.get_edge_data(v, u)
                                    if e_data:
                                        first_key = list(e_data.keys())[0]
                                        edge_types.append(e_data[first_key].get("type", "ASSOCIATED_WITH"))
                                    else:
                                        edge_types.append("ASSOCIATED_WITH")
                                        
                                score_details = self.calculate_path_score(path, edge_types)
                                
                                discovered_paths.append({
                                    "path": path,
                                    "labels": ["Material", "Property", dev_mid_label, "Device"],
                                    "relations": edge_types,
                                    "metrics": score_details
                                })
                except nx.NetworkXNoPath:
                    continue
                    
        discovered_paths.sort(key=lambda x: x["metrics"]["score"], reverse=True)
        return discovered_paths

    def get_all_discoveries(self):
        """
        Executes all active rules to find interdisciplinary bridging paths.
        """
        bridges = self.discover_customer_to_machine_bridges()
        materials_bridges = self.discover_material_to_device_bridges()
        
        return bridges + materials_bridges
