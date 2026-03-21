"""
PDF Generator Service
=====================

Converts summary text into a styled, downloadable PDF using fpdf2.
"""

from io import BytesIO
from datetime import datetime

from fpdf import FPDF


class SummaryPDFGenerator:
    """Generates a formatted PDF from summary text."""

    def __init__(self) -> None:
        self._pdf: FPDF | None = None

    def generate(
        self,
        summary: str,
        level_name: str,
        filename: str,
    ) -> BytesIO:
        """Create a PDF document and return it as a BytesIO stream.

        Args:
            summary: The summary text to render.
            level_name: Name of the summary level (e.g. "Executive").
            filename: Original PDF filename for the header.

        Returns:
            BytesIO buffer containing the PDF bytes.
        """
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=20)
        pdf.add_page()

        # --- Title ---
        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 12, f"{level_name} Summary", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(4)

        # --- Metadata line ---
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 100, 100)
        meta = f"Source: {filename}  |  Generated: {datetime.now().strftime('%d %b %Y, %H:%M')}"
        pdf.cell(0, 8, meta, new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(2)

        # --- Divider ---
        pdf.set_draw_color(200, 200, 200)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(8)

        # --- Body ---
        pdf.set_text_color(30, 30, 30)
        pdf.set_font("Helvetica", "", 11)

        for paragraph in summary.split("\n"):
            paragraph = paragraph.strip()
            if not paragraph:
                pdf.ln(4)
                continue

            # Check if it's a numbered extractive line
            if paragraph[0].isdigit() and ". [" in paragraph[:8]:
                pdf.set_font("Helvetica", "B", 10)
                pdf.multi_cell(0, 6, paragraph)
                pdf.set_font("Helvetica", "", 11)
            else:
                pdf.multi_cell(0, 6, paragraph)
            pdf.ln(2)

        # --- Footer word count ---
        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(130, 130, 130)
        word_count = len(summary.split())
        pdf.cell(0, 8, f"({word_count} words)", new_x="LMARGIN", new_y="NEXT", align="R")

        # Write to buffer
        buf = BytesIO()
        pdf.output(buf)
        buf.seek(0)
        return buf
