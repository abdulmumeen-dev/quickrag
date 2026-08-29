# QuickRAG

A lightweight, dependency-free document Q&A CLI tool. Point it at a folder 
of text files and ask questions — it retrieves the most relevant passages 
using TF-IDF similarity, no external API required.

## Features
- Zero heavy dependencies (pure Python standard library)
- Simple TF-IDF based semantic search
- Optional LLM integration for answer synthesis (set `ANTHROPIC_API_KEY`)
- Fast local indexing of `.txt` and `.md` files

## Usage
\`\`\`bash
python quickrag.py --docs ./my_documents --query "What is the refund policy?"
\`\`\`

## How it works
QuickRAG builds a TF-IDF index over your documents, ranks passages by 
cosine similarity to your query, and returns the most relevant matches. 
If an API key is set, it passes the retrieved context to an LLM for a 
synthesized answer.

## Roadmap
- [ ] Embedding-based retrieval (sentence-transformers)
- [ ] PDF/docx ingestion
- [ ] Persistent index caching
