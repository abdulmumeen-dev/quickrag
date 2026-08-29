#!/usr/bin/env python3
"""QuickRAG - Lightweight document Q&A using TF-IDF retrieval."""

import argparse
import math
import os
import re
import json
import urllib.request
from collections import Counter
from pathlib import Path


def tokenize(text):
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def load_documents(folder):
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix.lower() in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                docs.append({"source": str(path), "chunk_id": i, "text": chunk})
    return docs


def chunk_text(text, size=300, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += size - overlap
    return chunks if chunks else [text]


def build_tfidf(docs):
    tokenized = [tokenize(d["text"]) for d in docs]
    df = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    n_docs = len(docs)
    idf = {term: math.log(n_docs / (1 + freq)) + 1 for term, freq in df.items()}

    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
        vectors.append(vec)

    return vectors, idf


def vectorize_query(query, idf):
    tokens = tokenize(query)
    tf = Counter(tokens)
    total = len(tokens) or 1
    return {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}


def cosine_similarity(vec_a, vec_b):
    common = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve(query, docs, vectors, idf, top_k=3):
    q_vec = vectorize_query(query, idf)
    scored = [(cosine_similarity(q_vec, v), d) for v, d in zip(vectors, docs)]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [d for score, d in scored[:top_k] if score > 0]


def synthesize_answer(query, passages):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    context = "\n\n".join(p["text"] for p in passages)
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 500,
        "messages": [{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer using only the context above."
        }]
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
    )

    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read())
            return data["content"][0]["text"]
    except Exception as e:
        return f"(LLM synthesis failed: {e})"


def main():
    parser = argparse.ArgumentParser(description="QuickRAG - document Q&A CLI")
    parser.add_argument("--docs", required=True, help="Folder containing .txt/.md files")
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="Number of passages to retrieve")
    args = parser.parse_args()

    print(f"Indexing documents in {args.docs}...")
    docs = load_documents(args.docs)
    if not docs:
        print("No .txt or .md files found.")
        return

    vectors, idf = build_tfidf(docs)
    results = retrieve(args.query, docs, vectors, idf, top_k=args.top_k)

    if not results:
        print("No relevant passages found.")
        return

    print(f"\nTop {len(results)} relevant passages:\n")
    for r in results:
        print(f"--- {r['source']} (chunk {r['chunk_id']}) ---")
        print(r["text"][:300] + ("..." if len(r["text"]) > 300 else ""))
        print()

    answer = synthesize_answer(args.query, results)
    if answer:
        print("=== Synthesized Answer ===")
        print(answer)
    else:
        print("(Set ANTHROPIC_API_KEY to enable synthesized answers.)")


if __name__ == "__main__":
    main()
