"""
Neo4j Knowledge Graph with GraphRAG Integration
Provides relational source of truth with incremental updates.
"""
import os
from typing import List, Dict, Optional, Tuple, Set
from neo4j import GraphDatabase
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class Neo4jKnowledgeGraph:
    def __init__(
        self, 
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j"
    ):
        """
        Initialize Neo4j Knowledge Graph.
        
        Args:
            uri: Neo4j connection URI
            username: Database username
            password: Database password
            database: Database name
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.is_connected = False
        self.connection_error = None
        self._memory_entities = {}
        self._memory_relations = {}
        
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            self.is_connected = True
            logger.info("Connected to Neo4j successfully")
        except Exception as e:
            logger.warning(f"Neo4j connection failed: {e}. Running in offline mode.")
            self.driver = None
            self.is_connected = False
            self.connection_error = str(e)
    
    def close(self):
        """Close the connection."""
        if self.driver:
            self.driver.close()
    
    def _execute_query(self, query: str, params: Dict = None) -> List[Dict]:
        """Execute a Cypher query."""
        if not self.driver:
            logger.debug(f"[Offline Mode] Neo4j unavailable. Query not executed: {query[:50]}...")
            return []
        
        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(query, params or {})
                return [record.data() for record in result]
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            return []
    
    def add_entities(self, entities: List[Dict]) -> None:
        """
        Add entities to the knowledge graph.
        
        Args:
            entities: List of entity dicts with keys: name, label, description, source_papers
        """
        if not entities:
            return

        for entity in entities:
            name = entity.get("name", "").lower()
            if name:
                self._memory_entities[name] = {
                    "name": name,
                    "label": entity.get("label", "Unknown"),
                    "description": entity.get("description", ""),
                    "confidence": 1.0,
                }

        if not self.driver:
            return
        
        for entity in entities:
            query = """
            MERGE (e:Entity {name: $name})
            ON CREATE SET 
                e.label = $label,
                e.description = $description,
                e.created_at = datetime(),
                e.confidence = 1.0
            ON MATCH SET
                e.last_updated = datetime()
            RETURN e
            """
            
            params = {
                "name": entity.get("name", "").lower(),
                "label": entity.get("label", "Unknown"),
                "description": entity.get("description", "")
            }
            
            self._execute_query(query, params)
    
    def add_relations(self, relations: List[Dict]) -> None:
        """
        Add relationships between entities (incremental GraphRAG).
        
        Args:
            relations: List of relation dicts with keys: source, target, type, confidence, source_papers
        """
        if not relations:
            return

        for relation in relations:
            source = relation.get("source", "").lower()
            target = relation.get("target", "").lower()
            rel_type = relation.get("type", "RELATED_TO")
            confidence = relation.get("confidence", 0.75)
            key = (source, rel_type, target)
            existing = self._memory_relations.get(key)
            if existing:
                existing["confidence"] = (existing["confidence"] + confidence) / 2
                existing["evidence_count"] = existing.get("evidence_count", 1) + 1
            else:
                self._memory_relations[key] = {
                    "confidence": confidence,
                    "evidence_count": 1,
                }

        if not self.driver:
            return
        
        for relation in relations:
            query = """
            MATCH (s:Entity {name: $source})
            MATCH (t:Entity {name: $target})
            MERGE (s)-[r:RELATES {type: $type}]->(t)
            ON CREATE SET
                r.confidence = $confidence,
                r.evidence_count = 1,
                r.created_at = datetime()
            ON MATCH SET
                r.confidence = (r.confidence + $confidence) / 2,
                r.evidence_count = r.evidence_count + 1,
                r.last_updated = datetime()
            RETURN r
            """
            
            params = {
                "source": relation.get("source", "").lower(),
                "target": relation.get("target", "").lower(),
                "type": relation.get("type", "RELATED_TO"),
                "confidence": relation.get("confidence", 0.5)
            }
            
            self._execute_query(query, params)
    
    def find_reasoning_path(
        self, 
        start_entity: str, 
        end_entity: str, 
        max_depth: int = 5
    ) -> Optional[List[str]]:
        """
        Find a reasoning path between two entities using Dijkstra's algorithm.
        
        Args:
            start_entity: Starting entity name
            end_entity: Ending entity name
            max_depth: Maximum path depth
        
        Returns:
            List representing the path, or None if no path exists
        """
        if not self.driver:
            return None
        
        query = """
        MATCH (start:Entity {name: $start})
        MATCH (end:Entity {name: $end})
        CALL apoc.algo.dijkstra(start, end, 'RELATES>', 'confidence') 
            YIELD path, weight
        RETURN path, weight
        ORDER BY weight DESC
        LIMIT 1
        """
        
        params = {
            "start": start_entity.lower(),
            "end": end_entity.lower()
        }
        
        try:
            results = self._execute_query(query, params)
            if results:
                # Parse path from results
                path_data = results[0]
                return path_data.get("path", [])
        except Exception as e:
            logger.error(f"Path finding error: {e}")
        
        return None
    
    def get_entity_neighbors(self, entity_name: str, depth: int = 2) -> Dict:
        """
        Get all neighbors of an entity up to specified depth.
        
        Args:
            entity_name: Entity name
            depth: Traversal depth
        
        Returns:
            Dict with entity and its neighbors
        """
        if not self.driver:
            return {}
        
        query = f"""
        MATCH (e:Entity {{name: $name}})
        MATCH (e)-[r:RELATES*1..{depth}]-(neighbor:Entity)
        RETURN e, collect(DISTINCT neighbor) as neighbors, collect(DISTINCT r) as relations
        """
        
        params = {"name": entity_name.lower()}
        results = self._execute_query(query, params)
        
        if results:
            return results[0]
        return {}
    
    def get_knowledge_graph_stats(self) -> Dict:
        """Get statistics about the knowledge graph."""
        if not self.driver:
            return {}
        
        stats = {}
        
        # Count entities
        entity_query = "MATCH (e:Entity) RETURN count(e) as count"
        results = self._execute_query(entity_query)
        stats["total_entities"] = results[0]["count"] if results else 0
        
        # Count relationships
        rel_query = "MATCH ()-[r:RELATES]->() RETURN count(r) as count"
        results = self._execute_query(rel_query)
        stats["total_relationships"] = results[0]["count"] if results else 0
        
        # Get entity labels
        labels_query = "MATCH (e:Entity) RETURN DISTINCT e.label as label, count(e) as count"
        results = self._execute_query(labels_query)
        stats["entities_by_label"] = {r["label"]: r["count"] for r in results} if results else {}
        
        return stats
    
    def get_subgraph_for_query(self, query_entities: List[str], hops: int = 2) -> Dict:
        """
        Extract a subgraph relevant to given entities.
        
        Args:
            query_entities: List of entity names
            hops: Number of hops to traverse
        
        Returns:
            Dict with nodes and edges for visualization
        """
        if not self.driver or not query_entities:
            return {"nodes": [], "edges": []}
        
        # Escape entity names
        entity_names = [e.lower() for e in query_entities]
        placeholders = ", ".join([f"${i}" for i in range(len(entity_names))])
        
        query = f"""
        MATCH (start:Entity)
        WHERE start.name IN [{placeholders}]
        CALL apoc.path.expandConfig(start, {{
            relationshipFilter: 'RELATES',
            maxLevel: {hops},
            uniqueness: 'NODE_GLOBAL'
        }})
        YIELD path
        WITH nodes(path) as nodes, relationships(path) as rels
        UNWIND nodes as node
        UNWIND rels as rel
        RETURN 
            collect(DISTINCT {{id: id(node), name: node.name, label: node.label}}) as nodes,
            collect(DISTINCT {{source: id(startNode(rel)), target: id(endNode(rel)), type: type(rel)}}) as edges
        """
        
        params = {str(i): name for i, name in enumerate(entity_names)}
        
        try:
            results = self._execute_query(query, params)
            if results:
                return {
                    "nodes": results[0].get("nodes", []),
                    "edges": results[0].get("edges", [])
                }
        except Exception as e:
            logger.error(f"Subgraph extraction error: {e}")
        
        return {"nodes": [], "edges": []}
    
    def validate_fact(self, source: str, relation: str, target: str) -> Tuple[bool, float]:
        """
        Verify if a fact exists in the knowledge graph.
        
        Returns:
            Tuple of (exists, average_confidence)
        """
        if not self.driver:
            key = (source.lower(), relation, target.lower())
            stored = self._memory_relations.get(key)
            if stored:
                return True, stored.get("confidence", 0.75)
            return False, 0.0
        
        query = """
        MATCH (s:Entity {name: $source})-[r:RELATES {type: $type}]->(t:Entity {name: $target})
        RETURN r.confidence as confidence, r.evidence_count as evidence_count
        """
        
        params = {
            "source": source.lower(),
            "type": relation,
            "target": target.lower()
        }
        
        results = self._execute_query(query, params)
        
        if results:
            confidence = results[0].get("confidence", 0.0)
            return True, confidence
        
        return False, 0.0
    
    def incremental_update(self, new_entities: List[Dict], new_relations: List[Dict]) -> None:
        """
        Perform incremental GraphRAG update.
        
        Args:
            new_entities: New entities to add
            new_relations: New relations to add
        """
        self.add_entities(new_entities)
        self.add_relations(new_relations)
        logger.info(f"Incremental update: {len(new_entities)} entities, {len(new_relations)} relations")
