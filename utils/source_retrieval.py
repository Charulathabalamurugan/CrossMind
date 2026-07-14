import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

import requests


class SourceRetrievalError(RuntimeError):
    pass


_CACHE_TTL_SECONDS = 300
_CACHE: Dict[Tuple[str, int], Tuple[float, List[Dict]]] = {}


def _normalize_paper(paper: Dict, source: str) -> Dict:
    return {
        "paper_id": paper.get("paper_id") or f"{source}_{int(time.time() * 1000)}",
        "title": paper.get("title", "Untitled"),
        "abstract": paper.get("abstract", ""),
        "authors": paper.get("authors", ""),
        "year": paper.get("year", 2024),
        "citation_count": paper.get("citationCount", paper.get("citation_count", 0)),
        "venue": paper.get("venue", "Unknown Venue"),
        "source": source,
        "sources": [source],
    }


def aggregate_papers(papers: List[Dict], limit: int = 10) -> List[Dict]:
    merged: Dict[str, Dict] = {}
    for paper in papers:
        title = (paper.get("title") or "").strip().lower()
        abstract = (paper.get("abstract") or "").strip().lower()
        key = title or abstract or paper.get("paper_id") or ""
        if not key:
            continue
        if key not in merged:
            merged[key] = _normalize_paper(paper, paper.get("source", "unknown"))
        else:
            merged[key]["sources"] = sorted(set(merged[key]["sources"]) | {paper.get("source", "unknown")})
            merged[key]["source"] = merged[key]["sources"][0]
            if not merged[key].get("abstract") and paper.get("abstract"):
                merged[key]["abstract"] = paper.get("abstract")
            if not merged[key].get("authors") and paper.get("authors"):
                merged[key]["authors"] = paper.get("authors")
    return list(merged.values())[:limit]


def fetch_semantic_scholar(query: str, limit: int = 10) -> List[Dict]:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {"query": query, "limit": limit, "fields": "title,abstract,authors,year,citationCount,venue"}
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        items = response.json().get("data", [])
        return [_normalize_paper(item, "semantic_scholar") for item in items if item.get("abstract")]
    except Exception as exc:
        raise SourceRetrievalError(f"Semantic Scholar fetch failed: {exc}") from exc


def fetch_pubmed(query: str, limit: int = 10) -> List[Dict]:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmax": limit, "retmode": "json", "sort": "relevance"}
    try:
        search_response = requests.get(search_url, params=params, timeout=15)
        search_response.raise_for_status()
        ids = search_response.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_response = requests.get(
            fetch_url,
            params={"db": "pubmed", "id": ",".join(ids), "retmode": "xml", "rettype": "abstract"},
            timeout=15,
        )
        fetch_response.raise_for_status()

        root = ET.fromstring(fetch_response.text)
        papers = []
        for article in root.findall(".//PubmedArticle"):
            title = "".join(article.findtext(".//ArticleTitle", default="") or "")
            title = re.sub(r"\s+", " ", title).strip()
            abstract_parts = []
            for abstract_text in article.findall(".//AbstractText"):
                text = "".join(abstract_text.itertext())
                abstract_parts.append(text)
            abstract = " ".join(abstract_parts).strip()
            authors = []
            for author in article.findall(".//Author"):
                last = author.findtext("LastName", default="") or ""
                fore = author.findtext("Initials", default="") or ""
                if last or fore:
                    authors.append(f"{last} {fore}".strip())
            papers.append(
                _normalize_paper(
                    {
                        "paper_id": f"pubmed_{article.findtext('.//PMID', default='') or int(time.time()*1000)}",
                        "title": title or "Untitled PubMed article",
                        "abstract": abstract,
                        "authors": ", ".join(authors),
                        "year": 2024,
                        "citation_count": 0,
                        "venue": "PubMed",
                    },
                    "pubmed",
                )
            )
        return papers[:limit]
    except Exception as exc:
        raise SourceRetrievalError(f"PubMed fetch failed: {exc}") from exc


def fetch_ieee(query: str, limit: int = 10) -> List[Dict]:
    api_key = os.environ.get("IEEE_API_KEY")
    if not api_key:
        return []

    url = "https://ieeexploreapi.ieee.org/api/v1/search/articles"
    params = {"apikey": api_key, "querytext": query, "max_records": limit, "start_record": 1}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json().get("articles", [])
        papers = []
        for item in data:
            papers.append(
                _normalize_paper(
                    {
                        "paper_id": item.get("article_number") or item.get("doi") or f"ieee_{int(time.time()*1000)}",
                        "title": item.get("title", "Untitled IEEE article"),
                        "abstract": item.get("abstract"),
                        "authors": ", ".join([a.get("full_name", "") for a in item.get("authors", {}).get("authors", []) if a.get("full_name")]),
                        "year": item.get("publication_year") or 2024,
                        "citation_count": item.get("citation_count", 0),
                        "venue": item.get("publication_title", "IEEE"),
                    },
                    "ieee",
                )
            )
        return papers[:limit]
    except Exception as exc:
        raise SourceRetrievalError(f"IEEE Xplore fetch failed: {exc}") from exc


def fetch_arxiv(query: str, limit: int = 10) -> List[Dict]:
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": limit}
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.text)
        papers = []
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = " ".join(entry.find("{http://www.w3.org/2005/Atom}title").itertext()) if entry.find("{http://www.w3.org/2005/Atom}title") is not None else "Untitled arXiv article"
            title = re.sub(r"\s+", " ", title).strip()
            summary = " ".join(entry.find("{http://www.w3.org/2005/Atom}summary").itertext()) if entry.find("{http://www.w3.org/2005/Atom}summary") is not None else ""
            authors = [author.findtext("{http://www.w3.org/2005/Atom}name") or "" for author in entry.findall("{http://www.w3.org/2005/Atom}author")]
            papers.append(
                _normalize_paper(
                    {
                        "paper_id": entry.find("{http://www.w3.org/2005/Atom}id").text or f"arxiv_{int(time.time()*1000)}",
                        "title": title,
                        "abstract": summary,
                        "authors": ", ".join([a for a in authors if a]),
                        "year": 2024,
                        "citation_count": 0,
                        "venue": "arXiv",
                    },
                    "arxiv",
                )
            )
        return papers[:limit]
    except Exception as exc:
        raise SourceRetrievalError(f"arXiv fetch failed: {exc}") from exc


def clear_retrieval_cache() -> None:
    _CACHE.clear()


def fetch_multi_source_papers(query: str, limit: int = 10) -> List[Dict]:
    key = (query.lower().strip(), int(limit))
    now = time.time()
    if key in _CACHE:
        cached_at, cached_results = _CACHE[key]
        if now - cached_at <= _CACHE_TTL_SECONDS:
            return list(cached_results)

    sources = [fetch_semantic_scholar, fetch_pubmed, fetch_ieee, fetch_arxiv]
    all_papers: List[Dict] = []
    with ThreadPoolExecutor(max_workers=min(4, len(sources))) as executor:
        futures = {executor.submit(source_fn, query, limit=limit): source_fn.__name__ for source_fn in sources}
        for future in as_completed(futures):
            source_name = futures[future]
            try:
                all_papers.extend(future.result())
            except Exception as exc:
                print(f"{source_name} unavailable: {exc}")

    merged = aggregate_papers(all_papers, limit=limit)
    _CACHE[key] = (now, merged)
    return list(merged)
