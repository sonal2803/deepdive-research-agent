"""
DeepDive Insight Agent (GROQ + LLaMA-3 Version)
----------------------------------------------

Pipeline:
- Web search (Tavily)
- Web scraping
- Embeddings (SentenceTransformer)
- Clustering (KMeans)
- RAG-style grounding for a global answer
- Cluster-wise LLM analysis (Groq LLaMA-3)
"""

import requests
from bs4 import BeautifulSoup
from tavily import TavilyClient
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from groq import Groq
import numpy as np

# -------------------------------------------------
# Load API KEYS
# -------------------------------------------------
import streamlit as st

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
TAVILY_API_KEY = st.secrets["TAVILY_API_KEY"]


# -------------------------------------------------
# Groq Client + Model
# -------------------------------------------------
groq_client = Groq(api_key=GROQ_API_KEY)

# Recommended model: fast + good quality
LLM_MODEL = "llama-3.1-8b-instant"

# Tavily Search Client
tavily = TavilyClient(api_key=TAVILY_API_KEY)

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------
# 0. Small helpers
# ---------------------------------------------------------
def _is_low_quality(text: str) -> bool:
    """
    Filter out garbage pages:
    - Very short text
    - Obvious HTTP error / access issues
    """
    if not text:
        return True

    lowered = text.lower()
    if len(text) < 500:
        return True

    error_markers = [
        "403 forbidden",
        "404 not found",
        "access denied",
        "access is denied",
        "error code",
        "page you are looking for is temporarily unavailable",
    ]
    return any(m in lowered for m in error_markers)


def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 200):
    """
    Simple character-based chunking for RAG-style retrieval.
    """
    chunks = []
    text = text.strip()
    if not text:
        return chunks

    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks


# ---------------------------------------------------------
# 1. SEARCH THE WEB
# ---------------------------------------------------------
def search_web(query: str, n_results: int = 6):
    try:
        response = tavily.search(query=query, max_results=n_results)
        return response["results"]
    except Exception as e:
        print("Search error:", e)
        return []


# ---------------------------------------------------------
# 2. SCRAPE A WEBPAGE
# ---------------------------------------------------------
def scrape_page(url: str) -> str:
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        return soup.get_text(separator=" ", strip=True)

    except Exception as e:
        print("Scrape error:", e)
        return ""


# ---------------------------------------------------------
# 3. CLUSTER TEXTS
# ---------------------------------------------------------
def cluster_texts(text_list, k_clusters=3):
    """
    Cluster texts using sentence embeddings + KMeans.
    Returns a numpy array of cluster labels.
    """
    if not text_list:
        return np.array([])

    if len(text_list) < k_clusters:
        k_clusters = max(1, len(text_list))

    embeddings = embedder.encode(text_list)
    kmeans = KMeans(n_clusters=k_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    return labels


# ---------------------------------------------------------
# 4. LLM ANALYSIS USING GROQ + LLaMA-3 (cluster-level)
# ---------------------------------------------------------
def analyze_cluster(label: int, cluster_text: str):
    """
    Cluster-wise analysis.

    We explicitly ask the model to:
    - Summarize
    - Extract key insights
    - Highlight contradictions
    - Point out missing info
    - Give a conclusion
    """

    prompt = f"""
You are an expert research analyst.

You will analyze text that comes from a group of web pages 
which are all about a similar theme (a cluster).

==== CLUSTER {label} CONTENT ====
{cluster_text}
=================================

Follow THIS EXACT MARKDOWN FORMAT in your reply:

### Summary of Main Idea
- ...

### Key Insights
- ...

### Contradictions
- If different sources disagree, list them clearly.
- If there are no obvious contradictions, say "No clear contradictions found."

### Missing or Unclear Information
- ...

### Conclusion
- Write 2–4 sentences with your conclusion.
- Mention how reliable you think this cluster is (low / medium / high).
"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.35,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"GROQ Error: {str(e)}"


# ---------------------------------------------------------
# 4.b RAG-STYLE GLOBAL ANSWER (query-level)
# ---------------------------------------------------------
def build_rag_answer(query: str, pages, top_k: int = 10):
    """
    Simple RAG-style retrieval:
    - Chunk all pages
    - Embed chunks
    - Retrieve top_k chunks most similar to the query
    - Ask LLM to answer based only on those chunks
    """
    corpus_chunks = []

    for title, url, content in pages:
        if not content:
            continue
        for chunk in _chunk_text(content):
            corpus_chunks.append(
                {
                    "title": title,
                    "url": url,
                    "text": chunk,
                }
            )

    if not corpus_chunks:
        return "Not enough high-quality content to build a grounded answer."

    texts = [c["text"] for c in corpus_chunks]
    embeddings = embedder.encode(texts)
    query_emb = embedder.encode([query])[0]

    # cosine similarity
    dot = np.dot(embeddings, query_emb)
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    sims = dot / (norms + 1e-10)

    top_idx = sims.argsort()[-top_k:][::-1]
    evidence = [corpus_chunks[i] for i in top_idx]

    evidence_text = ""
    for i, ev in enumerate(evidence, start=1):
        snippet = ev["text"][:450].replace("\n", " ")
        evidence_text += (
            f"[{i}] {ev['title']} — {ev['url']}\n"
            f"Snippet: {snippet}\n\n"
        )

    prompt = f"""
You are a careful research assistant.

User question:
{query}

You are given evidence snippets from the web. Treat THESE as your only
trusted factual basis. If something is not clearly supported by them,
say that it is uncertain.

==== EVIDENCE SNIPPETS ====
{evidence_text}
===========================

Write a structured answer in MARKDOWN with:

### Direct Answer
A short, clear answer to the user's question.

### Evidence-Based Explanation
- Use bullet points.
- Refer to snippets with [1], [2], ... based on the list above.
- Explain how the evidence supports your answer.

### Contradictions or Uncertainty
- If snippets disagree or are weak / low quality, highlight that clearly.
- If there is no real contradiction, say so.

### Overall Confidence
- low / medium / high, with 1–2 sentences explaining why.
"""

    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.25,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"GROQ Error (RAG answer): {str(e)}"


# ---------------------------------------------------------
# 5. MAIN WORKFLOW
# ---------------------------------------------------------
def deep_research(query: str, n_results: int = 6, k_clusters: int = 3):
    """
    Full pipeline:
    - Search
    - Scrape + filter bad pages
    - RAG-style global answer
    - Cluster + cluster analyses
    - Build markdown report
    """

    # 1) Search
    results = search_web(query, n_results=n_results)
    if not results:
        return "No search results found. Try a different query."

    # 2) Scrape + filter
    pages = []
    text_corpus = []

    for r in results:
        url = r.get("url", "")
        title = r.get("title", "Untitled")
        if not url:
            continue

        print(f"\nScraping: {title} ({url})")
        content = scrape_page(url)

        # 🔍 Filter low-quality / error pages
        if _is_low_quality(content):
            print(f"Skipping low-quality page: {title} ({url})")
            continue

        pages.append((title, url, content))
        text_corpus.append(content)

    if not pages:
        return "All scraped pages were low-quality or errors. Try another query."

    # 3) RAG-style global answer
    rag_answer = build_rag_answer(query, pages, top_k=10)

    # 4) Clustering
    labels = cluster_texts(text_corpus, k_clusters=k_clusters)
    if labels.size == 0:
        return "Could not cluster content due to insufficient data."

    clusters = {}
    for label, page in zip(labels, pages):
        clusters.setdefault(int(label), []).append(page)

    # 5) Build report
    report = f"# 🔍 DeepDive Research Report\n\n**Query:** {query}\n\n---\n"

    # Global RAG answer at the top
    report += "## 🧠 RAG-Grounded Global Answer\n"
    report += rag_answer + "\n\n---\n"

    # Cluster sections
    for label in sorted(clusters.keys()):
        items = clusters[label]
        combined_text = "\n\n".join([i[2] for i in items])
        insight = analyze_cluster(label, combined_text)

        report += f"\n## 📌 Cluster {label}\n"
        report += insight + "\n\n"
        report += "### Sources:\n"

        for (title, url, _) in items:
            report += f"- **{title}** — {url}\n"

        report += "\n---\n"

    return report
