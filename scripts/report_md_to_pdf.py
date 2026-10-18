"""Minimal Markdown -> PDF converter using reportlab (no LaTeX/pandoc needed).

Supports the constructs used in docs/final_report.md: ATX headings,
paragraphs, bullet lists, pipe tables, bold/italic inline markup (rendered
as plain text with font switches for **bold**), and horizontal rules.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def inline(text: str) -> str:
    """Convert a small subset of inline markdown to reportlab markup."""
    text = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\w)\*(.+?)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r"<font face='Courier'>\1</font>", text)
    return text


def build(md_path: Path, pdf_path: Path) -> None:
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["Normal"], fontSize=9.5,
                          leading=13, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=16,
                        spaceBefore=10, spaceAfter=8, textColor=colors.HexColor("#1F3B57"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5,
                        spaceBefore=10, spaceAfter=5, textColor=colors.HexColor("#2C5F8A"))
    bullet = ParagraphStyle("bullet", parent=body, leftIndent=14, spaceAfter=3)

    story = []
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        if not line.strip():
            i += 1
            continue
        if line.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.6,
                                    color=colors.HexColor("#9BB4C8")))
            i += 1
            continue
        if line.startswith("# "):
            story.append(Paragraph(inline(line[2:]), h1))
            i += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), h2))
            i += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), h2))
            i += 1
            continue
        # pipe table
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            header = [c.strip() for c in line.strip().strip("|").split("|")]
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(
                    [inline(c.strip()) for c in lines[i].strip().strip("|").split("|")]
                )
                i += 1
            data = [[Paragraph(f"<b>{inline(c)}</b>", body) for c in header]]
            data += [[Paragraph(c, body) for c in r] for r in rows]
            tbl = Table(data, repeatRows=1)
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2C5F8A")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#9BB4C8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#EAF2FB")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(tbl)
            story.append(Spacer(1, 6))
            continue
        # bullet list
        if re.match(r"^\s*[*-]\s+", line):
            while i < len(lines) and re.match(r"^\s*[*-]\s+", lines[i]):
                item = re.sub(r"^\s*[*-]\s+", "", lines[i].rstrip())
                story.append(Paragraph("• " + inline(item), bullet))
                i += 1
            continue
        # numbered list
        if re.match(r"^\s*\d+\.\s+", line):
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                item = re.sub(r"^\s*\d+\.\s+", "", lines[i].rstrip())
                story.append(Paragraph(inline(item), bullet))
                i += 1
            continue
        # plain paragraph (may span multiple lines until blank)
        para = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not re.match(
            r"^(#|\||\s*[*-]\s|\s*\d+\.\s|---)", lines[i]
        ):
            para.append(lines[i].rstrip())
            i += 1
        story.append(Paragraph(inline(" ".join(para)), body))

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
        title=md_path.stem,
    )
    doc.build(story)


if __name__ == "__main__":
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/final_report.md")
    pdf = Path(sys.argv[2]) if len(sys.argv) > 2 else md.with_suffix(".pdf")
    build(md, pdf)
    print(f"wrote {pdf}")
