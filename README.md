# QuickRAG

A small, local-first document question-answering CLI. QuickRAG indexes a folder of notes, retrieves the most relevant passages with TF-IDF, and can optionally synthesize an answer through Anthropic's API. Retrieval works offline and has no required third-party dependencies.

## Features

- Search `.txt` and `.md` files with Python's standard library.
- Optional extraction from `.pdf` and `.docx` files.
- Overlapping word-based chunks and explainable cosine-similarity scores.
- A persistent JSON index under `.quickrag/index.json`; file additions, removals, and edits invalidate it automatically.
- Text output for people or `--json` output for scripts.
- Optional Anthropic answer synthesis; retrieved passages are still shown as citations.
- No API key is needed for indexing or retrieval.

## Requirements

- Python 3.10 or newer.
- For PDF and Word documents, install the optional parsers:

```bash
python -m pip install -r requirements-documents.txt
```

## Quick start

```bash
python quickrag.py --docs ./my_documents --query "What is the refund policy?"
```

QuickRAG creates its local cache inside the document folder. To rebuild every time, use `--no-cache`. To select another cache location, use `--cache-file ./index.json`.

### Useful options

```text
--top-k 5          Return up to five passages
--chunk-size 250   Set chunk length in words
--overlap 40       Set overlapping words between chunks
--no-answer        Retrieve only; never call an LLM
--json             Emit a JSON object, including passage scores and source paths
--no-cache         Rebuild the index without reading or writing a cache
```

A machine-readable example:

```bash
python quickrag.py --docs ./my_documents --query "How do I reset a password?" --json --no-answer
```

### Optional answer synthesis

Set an Anthropic API key in your environment to enable answer synthesis. QuickRAG never stores the key in its index or repository.

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="your-key"

# PowerShell
$env:ANTHROPIC_API_KEY = "your-key"
```

The optional model defaults to `claude-3-5-haiku-latest`; set `ANTHROPIC_MODEL` to choose a different model available to your account. Use `--no-answer` to guarantee that no network request is made.

## Tests

The test suite uses only Python's standard library:

```bash
python -m unittest discover -s tests -v
```

## Design notes and limitations

- Retrieval is keyword-based TF-IDF, not a vector database or semantic embedding model.
- Text extraction from scanned PDFs requires OCR, which is not included.
- Cache freshness is based on file size and modification time. Use `--no-cache` if an external process preserves both while changing file contents.
- The optional LLM receives retrieved passages only; avoid indexing documents you are not permitted to send to an external provider.

## Roadmap

- Optional embedding-based retrieval for paraphrases.
- OCR for scanned PDFs.
- Additional answer providers behind a small provider interface.
