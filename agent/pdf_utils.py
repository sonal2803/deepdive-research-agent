# agent/pdf_utils.py

import os
import re
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter

# -------------------------------------------------
# 1. Register Unicode font (DejaVuSans)
# -------------------------------------------------
FONT_PATH = os.path.join(os.path.dirname(__file__), "fonts", "DejaVuSans.ttf")

if os.path.exists(FONT_PATH):
    pdfmetrics.registerFont(TTFont("DejaVu", FONT_PATH))
    FONT_NAME = "DejaVu"
else:
    # Fallback if font missing (will support fewer characters)
    FONT_NAME = "Helvetica"


# -------------------------------------------------
# 2. Helper: strip emojis (ReportLab hates them)
# -------------------------------------------------
_EMOJI_PATTERN = re.compile(
    "["
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F700-\U0001F77F"
    u"\U0001F780-\U0001F7FF"
    u"\U0001F800-\U0001F8FF"
    u"\U0001F900-\U0001F9FF"
    u"\U0001FA00-\U0001FA6F"
    u"\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)


def _strip_emojis(text: str) -> str:
    return _EMOJI_PATTERN.sub("", text)


# -------------------------------------------------
# 3. Main function: build PDF and return BytesIO
# -------------------------------------------------
def generate_pdf(report_text: str) -> BytesIO:
    """
    Create a PDF from the full markdown/text report and
    return it as an in-memory BytesIO object.
    """

    clean_text = _strip_emojis(report_text)

    # SimpleDocTemplate can write to a buffer instead of a file
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)

    styles = getSampleStyleSheet()
    style = styles["Normal"]
    style.fontName = FONT_NAME
    style.leading = 14  # line spacing

    # Replace newlines with HTML line breaks for Paragraph
    story = [Paragraph(clean_text.replace("\n", "<br/>"), style)]

    doc.build(story)
    buffer.seek(0)  # go to start so Streamlit can read it

    return buffer
