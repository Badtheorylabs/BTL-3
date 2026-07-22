#!/usr/bin/env python3
"""Render the BTL-3 compression article and paper as polished PDFs."""

from __future__ import annotations

import html
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Frame,
)
from render_academic_paper import main as render_academic_paper


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf"
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"
ARIAL_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
ARIAL_ITALIC = "/System/Library/Fonts/Supplemental/Arial Italic.ttf"

NAVY = colors.HexColor("#172B4D")
BLUE = colors.HexColor("#356AE6")
TEAL = colors.HexColor("#19A28C")
GOLD = colors.HexColor("#E0A22B")
RED = colors.HexColor("#D65858")
INK = colors.HexColor("#172033")
MUTED = colors.HexColor("#5B677A")
LIGHT = colors.HexColor("#F3F5F8")
GRID = colors.HexColor("#D7DEE8")


def register_fonts() -> None:
    pdfmetrics.registerFont(TTFont("BTL", ARIAL))
    pdfmetrics.registerFont(TTFont("BTL-Bold", ARIAL_BOLD))
    pdfmetrics.registerFont(TTFont("BTL-Italic", ARIAL_ITALIC))
    pdfmetrics.registerFontFamily(
        "BTL", normal="BTL", bold="BTL-Bold", italic="BTL-Italic"
    )


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="BTL-Bold",
            fontSize=28,
            leading=32,
            textColor=NAVY,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["Heading2"],
            fontName="BTL",
            fontSize=15,
            leading=20,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="BTL-Bold",
            fontSize=20,
            leading=24,
            textColor=NAVY,
            spaceBefore=16,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="BTL-Bold",
            fontSize=14.5,
            leading=18,
            textColor=NAVY,
            spaceBefore=13,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "H3",
            parent=base["Heading3"],
            fontName="BTL-Bold",
            fontSize=11.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="BTL",
            fontSize=9.4,
            leading=13.2,
            textColor=INK,
            spaceAfter=6,
            splitLongWords=True,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="BTL",
            fontSize=7.5,
            leading=9.5,
            textColor=MUTED,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="BTL-Italic",
            fontSize=10.5,
            leading=14.5,
            textColor=NAVY,
            leftIndent=14,
            rightIndent=10,
            borderColor=TEAL,
            borderWidth=2,
            borderPadding=(7, 9, 7, 10),
            backColor=colors.HexColor("#EDF8F6"),
            spaceBefore=7,
            spaceAfter=9,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.5,
            leading=10,
            textColor=INK,
            backColor=LIGHT,
            borderColor=GRID,
            borderWidth=0.5,
            borderPadding=7,
            leftIndent=5,
            rightIndent=5,
            spaceBefore=5,
            spaceAfter=8,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="BTL-Italic",
            fontSize=7.8,
            leading=10,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceBefore=3,
            spaceAfter=9,
        ),
    }


def inline(text: str) -> str:
    value = html.escape(text, quote=False)
    value = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', value)
    value = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", value)
    value = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", value)
    value = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r'<link href="\2" color="#356AE6">\1</link>',
        value,
    )
    value = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", value)
    return value


def page_header(canvas, doc) -> None:
    canvas.saveState()
    width, height = A4
    if doc.page > 1:
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(18 * mm, height - 15 * mm, width - 18 * mm, height - 15 * mm)
        canvas.setFont("BTL-Bold", 8)
        canvas.setFillColor(NAVY)
        canvas.drawString(18 * mm, height - 11.5 * mm, "BAD THEORY LABS")
        canvas.setFont("BTL", 8)
        canvas.setFillColor(MUTED)
        canvas.drawRightString(
            width - 18 * mm, height - 11.5 * mm, doc.running_title
        )
    canvas.setFont("BTL", 8)
    canvas.setFillColor(MUTED)
    canvas.drawCentredString(width / 2, 10 * mm, str(doc.page))
    canvas.restoreState()


class ResearchDoc(BaseDocTemplate):
    def __init__(self, filename: Path, running_title: str):
        super().__init__(
            str(filename),
            pagesize=A4,
            leftMargin=19 * mm,
            rightMargin=19 * mm,
            topMargin=21 * mm,
            bottomMargin=17 * mm,
            title=running_title,
            author="Bad Theory Labs",
            subject="BTL-3 compression research",
        )
        self.running_title = running_title
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
        )
        self.addPageTemplates(
            PageTemplate(id="main", frames=[frame], onPage=page_header)
        )


def title_block(
    title: str,
    subtitle: str,
    strapline: str,
    metric: str,
    metric_label: str,
    st: dict[str, ParagraphStyle],
) -> list[Flowable]:
    cards = Table(
        [
            [
                Paragraph("<b>8.39 GB</b><br/><font size='8'>native GGUF</font>", st["body"]),
                Paragraph("<b>27B class</b><br/><font size='8'>text agent</font>", st["body"]),
                Paragraph(f"<b>{metric}</b><br/><font size='8'>{metric_label}</font>", st["body"]),
            ]
        ],
        colWidths=[52 * mm, 52 * mm, 52 * mm],
        rowHeights=[24 * mm],
    )
    cards.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.7, GRID),
                ("INNERGRID", (0, 0), (-1, -1), 0.7, GRID),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TEXTCOLOR", (0, 0), (-1, -1), NAVY),
            ]
        )
    )
    return [
        Spacer(1, 18 * mm),
        Paragraph("BAD THEORY LABS / TECHNICAL RECORD", st["small"]),
        Spacer(1, 5 * mm),
        Paragraph(inline(title), st["title"]),
        Paragraph(inline(subtitle), st["subtitle"]),
        Spacer(1, 7 * mm),
        cards,
        Spacer(1, 12 * mm),
        Paragraph(inline(strapline), st["quote"]),
        Spacer(1, 7 * mm),
        Paragraph(
            "July 2026  |  BTL-3  |  text-only CUDA and Metal artifact",
            st["small"],
        ),
        PageBreak(),
    ]


def markdown_table(lines: list[str], st: dict[str, ParagraphStyle]) -> Table:
    rows: list[list[Paragraph]] = []
    for index, line in enumerate(lines):
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if index == 1 and all(set(cell) <= set("-:") for cell in cells):
            continue
        style = st["small"]
        rows.append([Paragraph(inline(cell), style) for cell in cells])
    widths = [None] * len(rows[0])
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "BTL-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def image_flow(path_text: str, caption: str, st: dict[str, ParagraphStyle]) -> list[Flowable]:
    path = ROOT / path_text
    if not path.is_file():
        raise FileNotFoundError(path)
    max_width, max_height = 168 * mm, 105 * mm
    item = Image(str(path))
    scale = min(max_width / item.imageWidth, max_height / item.imageHeight)
    item.drawWidth = item.imageWidth * scale
    item.drawHeight = item.imageHeight * scale
    return [Spacer(1, 4 * mm), item, Paragraph(inline(caption), st["caption"])]


def is_special(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or stripped.startswith("|")
        or stripped.startswith("- ")
        or re.match(r"^\d+\. ", stripped) is not None
        or stripped.startswith(">")
        or stripped.startswith("```")
        or stripped.startswith("![")
        or stripped in {"---", "\\[", "\\]"}
    )


def parse_markdown(text: str, st: dict[str, ParagraphStyle],
                   skip_initial_titles: bool = True) -> list[Flowable]:
    lines = text.splitlines()
    story: list[Flowable] = []
    index = 0
    skipped_h1 = False
    while index < len(lines):
        raw = lines[index]
        line = raw.strip()
        if not line or line == "---":
            index += 1
            continue
        if line.startswith("# "):
            if skip_initial_titles and not skipped_h1:
                skipped_h1 = True
            else:
                story.append(Paragraph(inline(line[2:]), st["h1"]))
            index += 1
            continue
        if line.startswith("## "):
            story.append(Paragraph(inline(line[3:]), st["h1"]))
            index += 1
            continue
        if line.startswith("### "):
            story.append(Paragraph(inline(line[4:]), st["h2"]))
            index += 1
            continue
        image_match = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line)
        if image_match:
            story.extend(image_flow(image_match.group(2), image_match.group(1), st))
            index += 1
            continue
        if line.startswith("|"):
            table_lines = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            story.extend([Spacer(1, 2 * mm), markdown_table(table_lines, st), Spacer(1, 3 * mm)])
            continue
        if line.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            index += 1
            story.append(Paragraph("<br/>".join(html.escape(x) for x in code_lines), st["code"]))
            continue
        if line == "\\[":
            equation = []
            index += 1
            while index < len(lines) and lines[index].strip() != "\\]":
                equation.append(lines[index].strip())
                index += 1
            index += 1
            story.append(Paragraph("<br/>".join(html.escape(x) for x in equation), st["code"]))
            continue
        if line.startswith(">"):
            quote = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip()[1:].strip())
                index += 1
            story.append(Paragraph(inline(" ".join(quote)), st["quote"]))
            continue
        if re.match(r"^\[\d+\] ", line):
            reference = [line]
            index += 1
            while index < len(lines) and lines[index].strip():
                if re.match(r"^\[\d+\] ", lines[index].strip()):
                    break
                reference.append(lines[index].strip())
                index += 1
            story.append(Paragraph(inline(" ".join(reference)), st["small"]))
            continue
        if line.startswith("- ") or re.match(r"^\d+\. ", line):
            ordered = not line.startswith("- ")
            items = []
            pattern = r"^(\d+)\. " if ordered else r"^- "
            while index < len(lines) and re.match(pattern, lines[index].strip()):
                match = re.match(pattern, lines[index].strip())
                item = re.sub(pattern, "", lines[index].strip())
                rendered = inline(item)
                rendered = f"<b>{match.group(1)}.</b> {rendered}" if ordered else rendered
                flow = Paragraph(rendered, st["body"])
                items.append(flow if ordered else ListItem(flow, leftIndent=8))
                index += 1
            if ordered:
                story.extend(items)
                continue
            story.append(
                ListFlowable(
                    items,
                    bulletType="bullet",
                    bulletChar="•",
                    leftIndent=16,
                    bulletFontName="BTL",
                    bulletFontSize=8,
                    spaceAfter=5,
                )
            )
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and not is_special(lines[index]):
            paragraph.append(lines[index].strip())
            index += 1
        story.append(Paragraph(inline(" ".join(paragraph)), st["body"]))
    return story


def render_article(st: dict[str, ParagraphStyle]) -> None:
    main = (ROOT / "ARTICLE.md").read_text()
    appendix = (ROOT / "ARTICLE-IMPLEMENTATION-APPENDIX.md").read_text()
    story = title_block(
        "Compressing BTL-3 to 8.39 GB",
        "How we stopped letting perplexity lie",
        "A novel behavior-first compression cookbook for preserving structured tool behavior in a 27B-class local agent.",
        "92.2%",
        "conditional tool retention",
        st,
    )
    story.extend(parse_markdown(main, st))
    story.extend(parse_markdown(appendix, st))
    ResearchDoc(
        OUT / "btl-3-compression-engineering-article.pdf",
        "BTL-3 compression engineering article",
    ).build(story)


def main() -> None:
    register_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    st = styles()
    render_article(st)
    render_academic_paper()
    print(f"wrote PDFs to {OUT}")


if __name__ == "__main__":
    main()
