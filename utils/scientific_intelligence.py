import os
import json
import requests
import networkx as nx
import re

# Custom Class for Scientific Intelligence Engine
def call_gemini_extraction(abstract_text, title, api_key, model="gemini-1.5-flash"):
    """
    Calls the Gemini API to perform zero-shot structured Entity & Relation Extraction.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    
    prompt = f"""
    You are the Scientific Intelligence Engine. Your task is to analyze the scientific paper abstract below and extract key scientific entities and their relationships.
    
    Paper Title: {title}
    Abstract: {abstract_text}
    
    Extract the following node categories:
    - CustomerNeed (e.g. user demands, adoption criteria, operators)
    - Obstacle (e.g. trust deficit, cognitive load, complexity, cost)
    - Mechanism (e.g. explainable AI, natural language interfaces, visualization)
    - Technology (e.g. reinforcement learning, intelligent machines, autonomous agents)
    - Material (e.g. Graphene, polymer)
    - Property (e.g. conductivity, flexibility)
    - Device (e.g. biosensor, pacemaker)
    
    Extract relationships using these verbs:
    - FACES_CHALLENGE
    - REQUIRES
    - IMPLEMENTS
    - MITIGATES
    - ENABLES
    - RESOLVES
    - HAS_PROPERTY
    - USED_IN
    - ASSOCIATED_WITH

    You must output a valid JSON object with the following structure. Do NOT include any markdown code blocks, backticks, or other text outside the JSON.
    {{
      "entities": [
        {{"name": "normalized_name_in_lowercase", "label": "Category", "description": "Short context description"}}
      ],
      "relations": [
        {{"source": "normalized_source_name", "target": "normalized_target_name", "type": "RELATION_TYPE"}}
      ]
    }}
    """
    
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
        if response.status_code != 200:
            print(f"Gemini API Error: {response.status_code} - {response.text}")
            return None

        result_json = response.json()
        candidates = result_json.get("candidates") or []
        if not candidates:
            print("Gemini extraction response missing candidates")
            return None

        text_response = candidates[0].get("content", {}).get("parts", [])[0].get("text", "")
        if not text_response:
            print("Gemini extraction returned empty text response")
            return None

        try:
            return json.loads(text_response.strip())
        except json.JSONDecodeError:
            print(f"Gemini extraction returned invalid JSON: {text_response}")
            return None
    except Exception as e:
        print(f"Failed to run Gemini extraction: {e}")
    return None

def extract_offline_heuristics(abstract_text, title):
    """
    SciBERT-inspired local heuristic parser. It scans the text for domain keywords
    and constructs scientific relations using dependency and keyword triggers.
    This runs offline with zero dependencies and zero latency.
    """
    text = (title + " " + abstract_text).lower()
    
    entities_map = {}
    relations = []
    
    # Define dictionary of target keywords and labels
    keyword_rules = {
        # Customer & Industry Road
        "customer": ("customer", "CustomerNeed", "End users or customers adopting automation"),
        "user": ("user", "CustomerNeed", "Operators or users of automated systems"),
        "operator": ("operator", "CustomerNeed", "System operators managing machines"),
        "cognitive complexity": ("cognitive_complexity", "Obstacle", "Mental overhead and complexity"),
        "cognitive load": ("cognitive_load", "Obstacle", "Operator mental load and stress"),
        "trust deficit": ("trust_deficit", "Obstacle", "Lack of trust in black-box systems"),
        "trust": ("trust", "Mechanism", "User trust in automated decisions"),
        "explainable ai": ("explainable_ai", "Mechanism", "AI systems providing explanations"),
        "xai": ("explainable_ai", "Mechanism", "Explainable AI architectures"),
        "intelligent machine": ("intelligent_machines", "Technology", "Machines driven by neural networks"),
        "intelligent machines": ("intelligent_machines", "Technology", "Machines driven by neural networks"),
        "autonomous agent": ("autonomous_agents", "Technology", "Independent AI actors"),
        "autonomous agents": ("autonomous_agents", "Technology", "Independent AI actors"),
        "automation": ("automation", "Technology", "Automated system workflows"),
        
        # General Scientific (Graphene/Sensors)
        "graphene": ("graphene", "Material", "Atomic sheet of carbon atoms"),
        "polymer": ("polymer", "Material", "Flexible compound macromolecule"),
        "conductivity": ("conductivity", "Property", "Electrical charge transmission efficiency"),
        "flexibility": ("flexibility", "Property", "Bending without breakage"),
        "sensor": ("sensor", "Device", "Physical state measurement device"),
        "biosensor": ("biosensor", "Device", "Biological state sensor"),
        "implant": ("implantable_devices", "Device", "Biocompatible surgical implants"),
        "biodegradable": ("biodegradable_plastics", "Material", "Plastics designed for microbial breakdown"),
        "plastic": ("polymer", "Material", "Synthetic polymer packaging materials"),
        "packaging": ("packaging", "Device", "Protective food and product containers"),
        "marine microbe": ("marine_microbes", "Mechanism", "Ocean-dwelling microbial degraders"),
        "marine microbes": ("marine_microbes", "Mechanism", "Ocean-dwelling microbial degraders"),
        "microbe": ("marine_microbes", "Mechanism", "Microbial biodegradation agents"),
    }
    
    # Find matching entities
    for key, (name, label, desc) in keyword_rules.items():
        if key in text:
            entities_map[name] = {"name": name, "label": label, "description": desc}
            
    # Establish relations based on proximity triggers
    def add_rel(src, tgt, rel_type):
        if src in entities_map and tgt in entities_map:
            relations.append({"source": src, "target": tgt, "type": rel_type})

    # Customer & AI Relations
    if "cognitive" in text or "complexity" in text or "friction" in text or "load" in text:
        add_rel("customer", "cognitive_complexity", "FACES_CHALLENGE")
        add_rel("user", "cognitive_load", "FACES_CHALLENGE")
        add_rel("operator", "cognitive_load", "FACES_CHALLENGE")
        
    if "trust" in text and ("barrier" in text or "deficit" in text or "lack" in text):
        add_rel("customer", "trust_deficit", "FACES_CHALLENGE")
        add_rel("user", "trust_deficit", "FACES_CHALLENGE")
        
    if "explainable_ai" in entities_map:
        if "cognitive_complexity" in entities_map:
            add_rel("explainable_ai", "cognitive_complexity", "RESOLVES")
        if "cognitive_load" in entities_map:
            add_rel("explainable_ai", "cognitive_load", "RESOLVES")
        if "trust" in entities_map:
            add_rel("explainable_ai", "trust", "ENABLES")
            
    if "intelligent_machines" in entities_map:
        if "explainable_ai" in entities_map:
            add_rel("intelligent_machines", "explainable_ai", "REQUIRES")
        if "autonomous_agents" in entities_map:
            add_rel("intelligent_machines", "autonomous_agents", "ENABLES")
            
    # General Materials Relations
    if "graphene" in entities_map:
        if "conductivity" in entities_map:
            add_rel("graphene", "conductivity", "HAS_PROPERTY")
        if "flexibility" in entities_map:
            add_rel("graphene", "flexibility", "HAS_PROPERTY")
            
    if "sensor" in entities_map or "biosensor" in entities_map:
        device_name = "biosensor" if "biosensor" in entities_map else "sensor"
        if "graphene" in entities_map:
            add_rel("graphene", device_name, "USED_IN")
        if "implantable_devices" in entities_map:
            add_rel(device_name, "implantable_devices", "ASSOCIATED_WITH")

    # Biodegradable packaging & marine microbe relations
    if "biodegradable_plastics" in entities_map and "marine_microbes" in entities_map:
        add_rel("marine_microbes", "biodegradable_plastics", "ENABLES")
    if "polymer" in entities_map and "marine_microbes" in entities_map:
        add_rel("marine_microbes", "polymer", "ENABLES")
    if "packaging" in entities_map and "biodegradable_plastics" in entities_map:
        add_rel("biodegradable_plastics", "packaging", "USED_IN")
    if "graphene" in entities_map and "polymer" in entities_map:
        add_rel("graphene", "polymer", "ASSOCIATED_WITH")
    if "biosensor" in entities_map and "packaging" in entities_map:
        add_rel("biosensor", "packaging", "ASSOCIATED_WITH")
            
    # If no relations generated, add simple fallback relation to ensure graph connectedness
    if not relations and len(entities_map) >= 2:
        keys = list(entities_map.keys())
        add_rel(keys[0], keys[1], "ASSOCIATED_WITH")
        
    return {
        "entities": list(entities_map.values()),
        "relations": relations
    }

# Process papers to build the NetworkX graph
def construct_knowledge_graph(papers, api_key=None):
    """
    Builds a NetworkX graph from a collection of papers.
    For each paper, it extracts entities and relations (using Gemini or Local SciBERT heuristic)
    and merges them into a single coherent network.
    """
    G = nx.MultiDiGraph()
    
    for paper in papers:
        paper_id = paper["paper_id"]
        abstract = paper["abstract"]
        title = paper["title"]
        
        extracted = None
        if api_key:
            extracted = call_gemini_extraction(abstract, title, api_key)
            
        if not extracted:
            # Fall back to local SciBERT-inspired rule extraction
            extracted = extract_offline_heuristics(abstract, title)
            
        # Add nodes and edges to the graph
        for entity in extracted.get("entities", []):
            name = entity["name"]
            label = entity["label"]
            desc = entity["description"]
            
            # If node already exists, merge attributes
            if G.has_node(name):
                # Keep the label and append description
                existing_papers = G.nodes[name].get("paper_ids", [])
                if paper_id not in existing_papers:
                    G.nodes[name]["paper_ids"].append(paper_id)
            else:
                G.add_node(name, label=label, description=desc, paper_ids=[paper_id])
                
        for rel in extracted.get("relations", []):
            source = rel["source"]
            target = rel["target"]
            rel_type = rel["type"]
            
            # Only add edge if both nodes exist in the graph
            if G.has_node(source) and G.has_node(target):
                # Check if edge already exists with same type
                edge_exists = False
                for u, v, key, data in G.edges(keys=True, data=True):
                    if u == source and v == target and data.get("type") == rel_type:
                        if paper_id not in data.get("paper_ids", []):
                            data["paper_ids"].append(paper_id)
                        edge_exists = True
                        break
                if not edge_exists:
                    G.add_edge(source, target, type=rel_type, paper_ids=[paper_id])
                    
    return G
