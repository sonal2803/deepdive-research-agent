import sys
import os
from io import BytesIO

# -------------------------------------------------------------
# Fix Python import so Streamlit can locate 'agent'
# -------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# -------------------------------------------------------------
# Imports
# -------------------------------------------------------------
import streamlit as st
import pandas as pd

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from docx import Document
from docx.shared import Pt

from agent.research_agent import deep_research


# -------------------------------------------------------------
# Register DejaVu Font for Unicode PDF
# -------------------------------------------------------------
FONT_PATH = os.path.join(ROOT_DIR, "agent", "fonts", "DejaVuSans.ttf")

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont("DejaVu", FONT_PATH))
else:
    raise FileNotFoundError(
        f"Font file not found! Expected at: {FONT_PATH}\n"
        "Make sure DejaVuSans.ttf is placed as 'agent/fonts/DejaVuSans.ttf'."
    )


# -------------------------------------------------------------
# Helpers for exports
# -------------------------------------------------------------
def clean_text(text: str) -> str:
    """Remove emojis & characters that ReportLab cannot encode."""
    return "".join(ch for ch in text if ord(ch) < 65535)


def generate_pdf(report_text: str) -> bytes:
    """Generate a simple multi-page PDF using ReportLab (Unicode safe)."""
    buffer = BytesIO()

    c = canvas.Canvas(buffer, pagesize=letter)
    c.setFont("DejaVu", 11)

    width, height = letter
    margin = 40
    y = height - margin

    safe_text = clean_text(report_text)

    for line in safe_text.split("\n"):
        if y < margin:
            c.showPage()
            c.setFont("DejaVu", 11)
            y = height - margin

        c.drawString(margin, y, line)
        y -= 14

    c.save()
    buffer.seek(0)
    return buffer.read()


def generate_docx(report_text: str) -> bytes:
    """Generate a .docx version of the report."""
    doc = Document()
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    for line in report_text.split("\n"):
        doc.add_paragraph(line)

    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.read()


def compute_cluster_lengths(report_text: str) -> pd.DataFrame:
    """
    Rough analytics: estimate how much content belongs to each cluster
    by splitting the markdown on '## 📌 Cluster X' headings.
    """
    parts = report_text.split("\n## 📌 Cluster ")
    if len(parts) <= 1:
        # No clusters found
        return pd.DataFrame({"Cluster": [], "ContentLength": []})

    data = []
    # parts[0] is header + RAG answer section
    for part in parts[1:]:
        lines = part.splitlines()
        if not lines:
            continue

        # First line looks like '0', '1', '2' etc. (maybe with extra text)
        first_line = lines[0]
        cluster_id = first_line.split()[0].strip("#").strip()

        content = "\n".join(lines[1:])
        data.append({"Cluster": f"Cluster {cluster_id}", "ContentLength": len(content)})

    return pd.DataFrame(data)


# -------------------------------------------------------------
# Streamlit page config + custom CSS (UI polish)
# -------------------------------------------------------------
st.set_page_config(
    page_title="DeepDive AI Research Agent",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A bit of styling
st.markdown(
    """
    <style>
    /* Global background */
    .stApp {
        background-color: #050711;
        color: #f5f5f7;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Sidebar style */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111827, #020617);
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }

    /* Headings */
    h1, h2, h3 {
        color: #f9fafb;
    }

    /* Cards */
    .deep-card {
        background: radial-gradient(circle at top left, #111827, #020617);
        border-radius: 16px;
        padding: 18px 22px;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 35px rgba(15, 23, 42, 0.7);
        margin-bottom: 1.5rem;
    }

    .rag-header {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }

    .small-muted {
        font-size: 0.8rem;
        color: #9ca3af;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Sidebar controls
# -------------------------------------------------------------
st.sidebar.title("⚙️ DeepDive Controls")

n_results = st.sidebar.slider(
    "Number of web results",
    min_value=3,
    max_value=12,
    value=6,
    step=1,
)

k_clusters = st.sidebar.slider(
    "Number of clusters",
    min_value=2,
    max_value=6,
    value=3,
    step=1,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **This app:**
    - 🔎 Searches the web  
    - 🧹 Scrapes & filters low-quality pages  
    - 🧠 Clusters content  
    - 🤖 Uses Groq LLaMA-3 for analysis  
    - 📚 Produces a RAG-grounded global answer  
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<span class='small-muted'>Your API keys stay local on your machine.</span>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------
# Main layout
# -------------------------------------------------------------
st.title("🔍 DeepDive Research Report")

query = st.text_input(
    "Enter your research question:",
    placeholder="e.g. Is intermittent fasting scientifically proven?",
)

run_button = st.button("🚀 Run Deep Research", type="primary")

# We'll store the last report in session_state so the download
# buttons stay active without re-running the LLM call.
if "last_report" not in st.session_state:
    st.session_state["last_report"] = ""
    st.session_state["last_query"] = ""


if run_button:
    if query.strip() == "":
        st.warning("Please enter a query to research.")
    else:
        with st.spinner("Researching across the web, clustering, and analyzing..."):
            report = deep_research(
                query=query,
                n_results=n_results,
                k_clusters=k_clusters,
            )

        st.session_state["last_report"] = report
        st.session_state["last_query"] = query

# -------------------------------------------------------------
# Show report & tools if we have one
# -------------------------------------------------------------
if st.session_state["last_report"]:
    report = st.session_state["last_report"]
    query_used = st.session_state["last_query"]

    # ---------- Top summary / RAG box ----------
    with st.container():
        st.markdown(
            f"""
            <div class="deep-card">
                <div class="rag-header">🧠 RAG-Grounded Global Answer</div>
                <div class="small-muted">
                    Query: <code>{query_used}</code>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Show the full markdown report (which already contains the RAG answer + clusters)
    st.markdown(report)

    # ---------- Downloads row ----------
    st.markdown("### 📎 Export")

    col_pdf, col_docx = st.columns(2)

    with col_pdf:
        try:
            pdf_bytes = generate_pdf(report)
            st.download_button(
                label="📄 Download as PDF",
                data=pdf_bytes,
                file_name="DeepDive_Research_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"PDF generation failed: {e}")

    with col_docx:
        try:
            docx_bytes = generate_docx(report)
            st.download_button(
                label="📝 Download as Word (.docx)",
                data=docx_bytes,
                file_name="DeepDive_Research_Report.docx",
                mime=(
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"DOCX generation failed: {e}")

    # ---------- Simple analytics: cluster content length ----------
    st.markdown("### 📊 Cluster Content Overview")

    df_clusters = compute_cluster_lengths(report)
    if df_clusters.empty:
        st.caption("No cluster headings found in the report.")
    else:
        st.dataframe(df_clusters, use_container_width=True)
        st.bar_chart(
            df_clusters.set_index("Cluster")["ContentLength"],
            use_container_width=True,
        )

else:
    st.info("Enter a question and click **Run Deep Research** to start.")
