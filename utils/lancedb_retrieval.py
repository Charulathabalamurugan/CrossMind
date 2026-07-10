"""
LanceDB Hybrid Search System
Provides vector + BM25 search for efficient, scalable retrieval.
"""
import os
import json
import lancedb
import numpy as np
from typing import List, Dict, Optional, Tuple
from sentence_transformers import SentenceTransformer
from datetime import datetime

class LanceDBRetriever:
    def __init__(self, db_path: str = "./lancedb_store", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize LanceDB Vector Store with Hybrid Search capabilities.
        
        Args:
            db_path: Directory for LanceDB storage
            model_name: Sentence transformer model for embeddings
        """
        self.db_path = db_path
        os.makedirs(db_path, exist_ok=True)
        self.db = lancedb.connect(db_path)
        
        # Load embedding model
        self.embedding_model = SentenceTransformer(model_name)
        dim_fn = getattr(self.embedding_model, "get_embedding_dimension", None)
        if dim_fn:
            self.embedding_dim = dim_fn()
        else:
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Table names
        self.papers_table_name = "papers_vectors"
        self.triples_table_name = "knowledge_triples"
        
        self._initialize_tables()
    
    def _initialize_tables(self):
        """Create tables if they don't exist."""
        try:
            self.db.open_table(self.papers_table_name)
        except Exception:
            # Table doesn't exist yet
            pass
        
        try:
            self.db.open_table(self.triples_table_name)
        except Exception:
            # Table doesn't exist yet
            pass
    
    def add_papers(self, papers: List[Dict]) -> None:
        """
        Add papers to the vector store with embeddings.
        
        Args:
            papers: List of paper dicts with keys: paper_id, title, abstract, authors, year, citation_count, venue
        """
        if not papers:
            return
        
        # Create embeddings for abstracts
        abstracts = [p.get("abstract", "") for p in papers]
        embeddings = self.embedding_model.encode(abstracts, convert_to_numpy=True)
        
        # Prepare data for storage
        data = []
        for paper, embedding in zip(papers, embeddings):
            data.append({
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "abstract": paper.get("abstract", ""),
                "authors": paper.get("authors", ""),
                "year": paper.get("year", 0),
                "citation_count": paper.get("citation_count", 0),
                "venue": paper.get("venue", ""),
                "vector": embedding,
                "timestamp": datetime.now().isoformat()
            })
        
        # Add to table
        try:
            table = self.db.open_table(self.papers_table_name)
            table.add(data)
        except Exception:
            # Create new table
            self.db.create_table(self.papers_table_name, data=data, mode="overwrite")
    
    def add_knowledge_triples(self, triples: List[Dict]) -> None:
        """
        Add knowledge graph triples (graph RAG format).
        
        Args:
            triples: List of dicts with keys: source, relation, target, confidence, source_papers
        """
        if not triples:
            return
        
        # Create embeddings for triple text
        triple_texts = [
            f"{t.get('source', '')} {t.get('relation', '')} {t.get('target', '')}" 
            for t in triples
        ]
        embeddings = self.embedding_model.encode(triple_texts, convert_to_numpy=True)
        
        # Prepare data
        data = []
        for triple, embedding in zip(triples, embeddings):
            data.append({
                "source": triple.get("source", ""),
                "relation": triple.get("relation", ""),
                "target": triple.get("target", ""),
                "confidence": triple.get("confidence", 0.5),
                "source_papers": json.dumps(triple.get("source_papers", [])),
                "vector": embedding,
                "timestamp": datetime.now().isoformat()
            })
        
        # Add to table
        try:
            table = self.db.open_table(self.triples_table_name)
            table.add(data)
        except Exception:
            # Create new table
            self.db.create_table(self.triples_table_name, data=data, mode="overwrite")
    
    def hybrid_search_papers(
        self, 
        query: str, 
        k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3
    ) -> List[Dict]:
        """
        Perform hybrid search on papers (vector + BM25).
        
        Args:
            query: Search query
            k: Number of results to return
            vector_weight: Weight for vector search
            bm25_weight: Weight for BM25 search
        
        Returns:
            List of papers with hybrid scores
        """
        try:
            table = self.db.open_table(self.papers_table_name)
        except Exception:
            return []
        
        # Vector search
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        vector_results = table.search(query_embedding).limit(k * 2).to_list()
        
        # BM25 search (via LanceDB FTS) — falls back to vector-only when no FTS index exists
        bm25_results = []
        try:
            bm25_results = table.search(query, query_type="fts").limit(k * 2).to_list()
        except Exception as e:
            logger = __import__("logging").getLogger(__name__)
            logger.debug(f"FTS search unavailable, using vector results only: {e}")
        
        # Merge and score
        papers_dict = {}
        
        for i, result in enumerate(vector_results):
            paper_id = result.get("paper_id")
            vector_score = 1.0 - (i / len(vector_results))
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {"data": result, "vector_score": 0.0, "bm25_score": 0.0}
            papers_dict[paper_id]["vector_score"] = vector_score
        
        for i, result in enumerate(bm25_results):
            paper_id = result.get("paper_id")
            bm25_score = 1.0 - (i / len(bm25_results))
            if paper_id not in papers_dict:
                papers_dict[paper_id] = {"data": result, "vector_score": 0.0, "bm25_score": 0.0}
            papers_dict[paper_id]["bm25_score"] = bm25_score
        
        # Calculate hybrid scores
        results = []
        for paper_id, scores in papers_dict.items():
            hybrid_score = (
                vector_weight * scores["vector_score"] + 
                bm25_weight * scores["bm25_score"]
            )
            result = scores["data"]
            result["hybrid_score"] = hybrid_score
            results.append(result)
        
        # Sort by hybrid score and return top k
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        return results[:k]
    
    def hybrid_search_triples(
        self, 
        query: str, 
        k: int = 20,
        min_confidence: float = 0.0
    ) -> List[Dict]:
        """
        Search knowledge triples with high-confidence filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of knowledge triples
        """
        try:
            table = self.db.open_table(self.triples_table_name)
        except Exception:
            return []
        
        # Vector search
        query_embedding = self.embedding_model.encode(query, convert_to_numpy=True)
        results = table.search(query_embedding).limit(k * 2).to_list()
        
        # Filter by confidence
        filtered = [r for r in results if r.get("confidence", 0.0) >= min_confidence]
        
        # Parse source_papers JSON
        for result in filtered:
            if isinstance(result.get("source_papers"), str):
                result["source_papers"] = json.loads(result["source_papers"])
        
        return filtered[:k]
    
    def get_paper_by_id(self, paper_id: str) -> Optional[Dict]:
        """Retrieve a specific paper by ID."""
        try:
            table = self.db.open_table(self.papers_table_name)
            results = table.search().where(f"paper_id = '{paper_id}'").to_list()
            return results[0] if results else None
        except Exception:
            return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        stats = {}
        
        try:
            papers_table = self.db.open_table(self.papers_table_name)
            stats["total_papers"] = papers_table.count()
        except Exception:
            stats["total_papers"] = 0
        
        try:
            triples_table = self.db.open_table(self.triples_table_name)
            stats["total_triples"] = triples_table.count()
        except Exception:
            stats["total_triples"] = 0
        
        return stats
    
    def clear(self) -> None:
        """Clear all tables."""
        try:
            self.db.drop_table(self.papers_table_name)
        except Exception:
            pass
        
        try:
            self.db.drop_table(self.triples_table_name)
        except Exception:
            pass
