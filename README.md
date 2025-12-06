# DeepDive Research Agent

DeepDive is an AI-powered web research tool that:

* Searches the web using Tavily API
* Scrapes and cleans webpages
* Converts content into vector embeddings
* Clusters similar content using KMeans
* Analyzes each cluster using Groq LLaMA-3.1
* Generates a final RAG-style answer
* Detects contradictions, missing information, and conclusions
* Exports the research report as a PDF
* Provides an interactive Streamlit interface

---

## Features

* Web search (Tavily)
* Automatic webpage scraping and filtering
* Text embeddings (SentenceTransformer)
* KMeans clustering
* LLaMA-3.1 cluster-level analysis
* RAG-grounded global summary
* PDF report export
* Clean and responsive Streamlit UI

---

## Project Structure

```
deepdive-research-agent/
│── app/
│     └── app.py                 → Streamlit frontend
│── agent/
│     └── research_agent.py      → Core research engine
│── config/
│     └── keys.py                → Local API keys (ignored in Git)
│── .streamlit/
│     └── secrets.toml           → Streamlit Cloud secrets
│── requirements.txt
│── README.md
│── .gitignore
```

---

## API Keys (Important)

The project requires two API keys:

* Tavily API: [https://app.tavily.com](https://app.tavily.com)
* Groq API: [https://console.groq.com/keys](https://console.groq.com/keys)

### For Local Development

Create this file:

```
config/keys.py
```

Add:

```python
TAVILY_API_KEY = "your_tavily_key"
GROQ_API_KEY = "your_groq_key"
```

This file is excluded from Git via `.gitignore`.

---

## Streamlit Deployment

Streamlit Cloud does not use `keys.py`.
Instead, create this file in your repository:

```
.streamlit/secrets.toml
```

With:

```toml
[api_keys]
TAVILY_API_KEY = "your_tavily_key"
GROQ_API_KEY = "your_groq_key"
```

In your code, access secrets like:

```python
import streamlit as st

TAVILY_API_KEY = st.secrets["api_keys"]["TAVILY_API_KEY"]
GROQ_API_KEY = st.secrets["api_keys"]["GROQ_API_KEY"]
```

---

## Running Locally

Install dependencies:

```
pip install -r requirements.txt
```

Run the app:

```
streamlit run app/app.py
```

---

## Requirements

All dependencies are listed in `requirements.txt`, including:

* tavily-python
* beautifulsoup4
* requests
* sentence-transformers
* scikit-learn
* numpy
* streamlit
* fpdf
* groq
* python-docx
* reportlab

---

## PDF Export

DeepDive uses FPDF to generate a text-based downloadable PDF report summarizing the research.

---



