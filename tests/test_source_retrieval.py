import unittest
from unittest.mock import patch

from utils import source_retrieval


class SourceRetrievalTests(unittest.TestCase):
    def test_aggregate_sources_deduplicates_and_merges_results(self):
        papers = [
            {"paper_id": "p1", "title": "Graphene sensors", "abstract": "Graphene improves biosensors.", "source": "semantic_scholar"},
            {"paper_id": "p2", "title": "Graphene sensors", "abstract": "Graphene improves biosensors.", "source": "pubmed"},
            {"paper_id": "p3", "title": "Marine microbes packaging", "abstract": "Biodegradable packaging with microbes.", "source": "arxiv"},
        ]

        merged = source_retrieval.aggregate_papers(papers, limit=3)

        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0]["paper_id"], "p1")
        self.assertEqual(merged[0]["sources"], ["pubmed", "semantic_scholar"])
        self.assertEqual(merged[1]["source"], "arxiv")

    @patch("utils.source_retrieval.requests.get")
    def test_fetch_arxiv_parses_atom_feed(self, mock_get):
        xml_payload = """
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/1234.5678</id>
            <title>Deep learning for science</title>
            <summary>New results in scientific discovery.</summary>
            <author><name>Alice Example</name></author>
          </entry>
        </feed>
        """
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.text = xml_payload

        papers = source_retrieval.fetch_arxiv("science", limit=1)

        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]["title"], "Deep learning for science")
        self.assertIn("Alice Example", papers[0]["authors"])


if __name__ == "__main__":
    unittest.main()
