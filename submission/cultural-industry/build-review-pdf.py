#!/usr/bin/env python3
import json
import math
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, NextPageTemplate, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle
)

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "manuscript.json").read_text(encoding="utf-8"))
SERIF = Path("/tmp/kra-fonts/node_modules/@expo-google-fonts/noto-serif-kr/400Regular/NotoSerifKR_400Regular.ttf")
SERIF_BOLD = Path("/tmp/kra-fonts/node_modules/@expo-google-fonts/noto-serif-kr/700Bold/NotoSerifKR_700Bold.ttf")
SANS = Path("/tmp/kra-fonts/node_modules/@expo-google-fonts/noto-sans-kr/400Regular/NotoSansKR_400Regular.ttf")
SANS_BOLD = Path("/tmp/kra-fonts/node_modules/@expo-google-fonts/noto-sans-kr/700Bold/NotoSansKR_700Bold.ttf")
for name, file in [("NotoSerifKR", SERIF), ("NotoSerifKR-Bold", SERIF_BOLD), ("NotoSansKR", SANS), ("NotoSansKR-Bold", SANS_BOLD)]:
    pdfmetrics.registerFont(TTFont(name, str(file)))

PAGE_W, PAGE_H = LETTER
LEFT = RIGHT = 20 * mm
TOP = BOTTOM = 15 * mm
GAP = 8 * mm
COL_W = (PAGE_W - LEFT - RIGHT - GAP) / 2

class SubmissionDoc(BaseDocTemplate):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, pagesize=LETTER, leftMargin=LEFT, rightMargin=RIGHT, topMargin=TOP, bottomMargin=BOTTOM, **kwargs)
        full = Frame(LEFT, BOTTOM, PAGE_W - LEFT - RIGHT, PAGE_H - TOP - BOTTOM, id="full")
        left = Frame(LEFT, BOTTOM, COL_W, PAGE_H - TOP - BOTTOM, id="left", rightPadding=2*mm)
        right = Frame(LEFT + COL_W + GAP, BOTTOM, COL_W, PAGE_H - TOP - BOTTOM, id="right", leftPadding=2*mm)
        self.addPageTemplates([
            PageTemplate(id="Front", frames=[full], onPage=self._page),
            PageTemplate(id="Body", frames=[left, right], onPage=self._page),
        ])
    def _page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("NotoSerifKR", 8)
        canvas.drawCentredString(PAGE_W/2, 8*mm, str(doc.page))
        canvas.restoreState()

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("title", fontName="NotoSansKR-Bold", fontSize=17.5, leading=27, alignment=TA_CENTER, spaceAfter=7*mm),
    "en_title": ParagraphStyle("en_title", fontName="NotoSansKR-Bold", fontSize=11.5, leading=16, alignment=TA_CENTER, spaceAfter=4*mm),
    "author": ParagraphStyle("author", fontName="NotoSansKR-Bold", fontSize=10, leading=16, alignment=TA_CENTER, spaceAfter=5*mm),
    "abstract_head": ParagraphStyle("abstract_head", fontName="NotoSerifKR-Bold", fontSize=9.5, leading=15, alignment=TA_CENTER, spaceBefore=2*mm, spaceAfter=1*mm),
    "front": ParagraphStyle("front", fontName="NotoSerifKR", fontSize=8.4, leading=13.4, alignment=TA_JUSTIFY, firstLineIndent=3*mm, wordWrap="CJK"),
    "major": ParagraphStyle("major", fontName="NotoSansKR-Bold", fontSize=13.5, leading=20, alignment=TA_LEFT, spaceBefore=4*mm, spaceAfter=3*mm, keepWithNext=True),
    "minor": ParagraphStyle("minor", fontName="NotoSansKR-Bold", fontSize=10.5, leading=16, alignment=TA_LEFT, leftIndent=3*mm, spaceBefore=3*mm, spaceAfter=2*mm, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="NotoSerifKR", fontSize=8.8, leading=14.1, alignment=TA_JUSTIFY, firstLineIndent=3*mm, spaceAfter=1.2*mm, wordWrap="CJK", splitLongWords=True),
    "note": ParagraphStyle("note", fontName="NotoSerifKR", fontSize=7.6, leading=10.5, alignment=TA_JUSTIFY, wordWrap="CJK", spaceAfter=1.5*mm),
    "table_title": ParagraphStyle("table_title", fontName="NotoSansKR-Bold", fontSize=8.6, leading=11, alignment=TA_LEFT, spaceBefore=2*mm, spaceAfter=1*mm, keepWithNext=True),
    "table_cell": ParagraphStyle("table_cell", fontName="NotoSansKR", fontSize=6.5, leading=8.2, alignment=TA_LEFT, wordWrap="CJK"),
    "table_head": ParagraphStyle("table_head", fontName="NotoSansKR-Bold", fontSize=6.5, leading=8.2, alignment=TA_CENTER, wordWrap="CJK"),
    "ref": ParagraphStyle("ref", fontName="NotoSerifKR", fontSize=8.5, leading=13.6, alignment=TA_JUSTIFY, leftIndent=5*mm, firstLineIndent=-5*mm, spaceAfter=1.2*mm, wordWrap="CJK"),
}

def P(text, style="body"):
    safe = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace("\n", "<br/>"))
    return Paragraph(safe, S[style])

def table_flow(t):
    rows = [[P(x, "table_head") for x in t["headers"]]] + [[P(x, "table_cell") for x in row] for row in t["rows"]]
    if len(t["headers"]) == 3:
        widths = [COL_W*0.18, COL_W*0.34, COL_W*0.48]
    else:
        widths = [COL_W*0.14, COL_W*0.17, COL_W*0.20, COL_W*0.245, COL_W*0.245]
    tab = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    tab.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), "NotoSansKR"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("ALIGN", (0,1), (0,-1), "CENTER"),
        ("LINEABOVE", (0,0), (-1,0), 1.2, colors.black),
        ("LINEBELOW", (0,0), (-1,0), 0.5, colors.black),
        ("LINEBELOW", (0,-1), (-1,-1), 1.2, colors.black),
        ("LEFTPADDING", (0,0), (-1,-1), 1.5),
        ("RIGHTPADDING", (0,0), (-1,-1), 1.5),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    return [P(t["title"], "table_title"), tab, P(t["note"], "note")]

def make_pdf(path, anonymous=False):
    story = [P(DATA["title"], "title"), P(DATA["englishTitle"], "en_title")]
    if not anonymous:
        story += [P(f'{DATA["author"]} ({DATA["englishAuthor"]})', "author"), P(f'{DATA["affiliation"]} · {DATA["email"]}', "note")]
    story += [P("초록", "abstract_head"), P(DATA["abstract"], "front"), Spacer(1, 2*mm), P(f'핵심어: {DATA["keywords"]}', "front"),
              Spacer(1, 4*mm), P("Abstract", "abstract_head"), P(DATA["englishAbstract"], "front"), Spacer(1, 2*mm), P(f'Keywords: {DATA["englishKeywords"]}', "front"),
              NextPageTemplate("Body"), PageBreak()]
    note = "편집위원회 참고: 익명 심사용 원고" if anonymous else DATA["technicalNote"]
    story += [P(note, "note")]
    for sec in DATA["sections"]:
        story.append(P(sec["heading"], "major"))
        story.extend(P(x) for x in sec.get("paragraphs", []))
        for sub in sec.get("subsections", []):
            story.append(P(sub["heading"], "minor"))
            story.extend(P(x) for x in sub.get("paragraphs", []))
            if "table" in sub:
                story.extend(table_flow(sub["table"]))
            story.extend(P(x) for x in sub.get("afterTable", []))
    story += [P("데이터와 코드 가용성", "minor"), P(DATA["availability"]), P("참고문헌", "major")]
    story.extend(P(x, "ref") for x in DATA["references"])
    doc = SubmissionDoc(str(path), title=DATA["title"], author="" if anonymous else DATA["author"])
    doc.build(story)

make_pdf(HERE / "kra-cultural-industry-submission-author-review.pdf", anonymous=False)
make_pdf(HERE / "kra-cultural-industry-submission-blind-review.pdf", anonymous=True)
print("review PDFs generated")
