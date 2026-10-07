import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import quickrag


class ChunkingTests(unittest.TestCase):
    def test_empty_text_has_no_chunks(self):
        self.assertEqual(quickrag.chunk_text("  \n "), [])

    def test_chunks_respect_size_and_overlap(self):
        chunks = quickrag.chunk_text("one two three four five six", size=4, overlap=1)
        self.assertEqual(chunks, ["one two three four", "four five six"])

    def test_invalid_chunk_settings_fail_early(self):
        with self.assertRaises(ValueError):
            quickrag.chunk_text("text", size=0)
        with self.assertRaises(ValueError):
            quickrag.chunk_text("text", size=2, overlap=2)


class RetrievalTests(unittest.TestCase):
    def test_query_ranks_relevant_passage_first(self):
        docs = [
            {"source": "billing.md", "chunk_id": 0, "text": "Invoices are due within thirty days."},
            {"source": "garden.md", "chunk_id": 0, "text": "Tomatoes need sunlight and regular watering."},
        ]
        vectors, idf = quickrag.build_tfidf(docs)
        results = quickrag.retrieve("when are invoices due", docs, vectors, idf, top_k=2)
        self.assertEqual(results[0]["source"], "billing.md")
        self.assertGreater(results[0]["score"], 0)

    def test_unknown_terms_return_no_matches(self):
        docs = [{"source": "a.txt", "chunk_id": 0, "text": "known words"}]
        vectors, idf = quickrag.build_tfidf(docs)
        self.assertEqual(quickrag.retrieve("unseen token", docs, vectors, idf), [])


class IndexTests(unittest.TestCase):
    def test_cache_reused_and_invalidated_after_document_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text("Refunds are processed within five days.", encoding="utf-8")
            first = quickrag.load_or_build_index(root)
            self.assertFalse(first[3])
            self.assertTrue((root / ".quickrag" / "index.json").is_file())

            second = quickrag.load_or_build_index(root)
            self.assertTrue(second[3])
            self.assertEqual(first[:3], second[:3])

            (root / "guide.md").write_text("Refunds take seven working days to process.", encoding="utf-8")
            third = quickrag.load_or_build_index(root)
            self.assertFalse(third[3])
            self.assertIn("seven", third[2])

    def test_hidden_files_and_cache_are_not_ingested(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "public.txt").write_text("public document", encoding="utf-8")
            hidden = root / ".private"
            hidden.mkdir()
            (hidden / "secret.txt").write_text("secret document", encoding="utf-8")
            docs = quickrag.load_documents(root)
            self.assertEqual([doc["source"] for doc in docs], ["public.txt"])

    def test_malformed_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "doc.txt").write_text("cache recovery", encoding="utf-8")
            cache = root / "cache.json"
            cache.write_text("not-json", encoding="utf-8")
            docs, vectors, idf, hit = quickrag.load_or_build_index(root, cache_file=cache)
            self.assertFalse(hit)
            self.assertEqual(len(docs), len(vectors))
            self.assertIn("cache", idf)
            json.loads(cache.read_text(encoding="utf-8"))


class CliTests(unittest.TestCase):
    def test_json_output_contains_citation_and_cache_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "policy.txt").write_text("Returns are accepted within fourteen days.", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = quickrag.main(["--docs", str(root), "--query", "returns accepted", "--json", "--no-answer"])
            self.assertEqual(status, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["passages"][0]["source"], "policy.txt")
            self.assertFalse(result["cache_hit"])

    def test_missing_documents_directory_returns_error(self):
        output = io.StringIO()
        with contextlib.redirect_stderr(output):
            status = quickrag.main(["--docs", "/path/that/does/not/exist", "--query", "anything"])
        self.assertEqual(status, 2)
        self.assertIn("does not exist", output.getvalue())


if __name__ == "__main__":
    unittest.main()
