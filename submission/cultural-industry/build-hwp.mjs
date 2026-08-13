import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { initSync, HwpDocument } from '@rhwp/core';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../..');
const data = JSON.parse(fs.readFileSync(path.join(here, 'manuscript.json'), 'utf8'));
const templatePath = process.env.CULTURAL_INDUSTRY_TEMPLATE;
if (!templatePath) throw new Error('CULTURAL_INDUSTRY_TEMPLATE is required');
const wasmPath = new URL('./node_modules/@rhwp/core/rhwp_bg.wasm', import.meta.url);
initSync({ module: fs.readFileSync(wasmPath) });

globalThis.measureTextWidth = (font, value) => {
  const match = String(font).match(/([0-9.]+)px/);
  const size = match ? Number(match[1]) : 12;
  let units = 0;
  for (const ch of String(value)) units += ch.codePointAt(0) > 0x7f ? 0.92 : (ch === ' ' ? 0.32 : 0.53);
  return units * size;
};

const BORDER_NONE = { type: 0, width: 0, color: '#000000' };
const BORDER_THIN = { type: 1, width: 1, color: '#000000' };
const BORDER_THICK = { type: 1, width: 4, color: '#000000' };

const bodyChar = { fontFamily: '휴먼명조', fontSize: 900, ratios: [90,90,90,90,90,90,90], spacings: [-5,-5,-5,-5,-5,-5,-5] };
const frontBodyChar = { fontFamily: '휴먼명조', fontSize: 850, ratios: [90,90,90,90,90,90,90], spacings: [-5,-5,-5,-5,-5,-5,-5] };
const titleChar = { fontFamily: 'HY견고딕', fontSize: 1800, bold: true, ratios: [98,98,98,98,98,98,98], spacings: [-2,-2,-2,-2,-2,-2,-2] };
const authorChar = { fontFamily: '한컴돋움', fontSize: 1000, bold: true, ratios: [98,98,98,98,98,98,98], spacings: [-2,-2,-2,-2,-2,-2,-2] };
const majorChar = { fontFamily: 'HY견고딕', fontSize: 1400, ratios: [98,98,98,98,98,98,98], spacings: [-2,-2,-2,-2,-2,-2,-2] };
const minorChar = { fontFamily: 'HY견고딕', fontSize: 1100, ratios: [95,95,95,95,95,95,95], spacings: [-5,-5,-5,-5,-5,-5,-5] };
const tableChar = { fontFamily: '중고딕', fontSize: 700, ratios: [80,80,80,80,80,80,80], spacings: [-10,-10,-10,-10,-10,-10,-10] };
const tableHeadChar = { ...tableChar, bold: true };
const tableTitleChar = { fontFamily: '중고딕', fontSize: 900, bold: true, ratios: [80,80,80,80,80,80,80], spacings: [-10,-10,-10,-10,-10,-10,-10] };
const noteChar = { fontFamily: '휴먼명조', fontSize: 800, ratios: [90,90,90,90,90,90,90], spacings: [-5,-5,-5,-5,-5,-5,-5] };
const refChar = { fontFamily: '휴먼명조', fontSize: 900, ratios: [90,90,90,90,90,90,90], spacings: [-5,-5,-5,-5,-5,-5,-5] };

const pBody = { alignment: 'justify', lineSpacing: 160, indent: 13.3, spacingBefore: 0, spacingAfter: 0, keepLines: false, widowOrphan: true };
const pMajor = { alignment: 'left', lineSpacing: 180, indent: 0, spacingBefore: 500, spacingAfter: 1400, keepWithNext: true };
const pMinor = { alignment: 'left', lineSpacing: 180, indent: 13.3, spacingBefore: 300, spacingAfter: 800, keepWithNext: true };

function ok(value, label) {
  const parsed = typeof value === 'string' ? JSON.parse(value) : value;
  if (parsed && parsed.ok === false) throw new Error(`${label}: ${JSON.stringify(parsed)}`);
  return parsed;
}

function controls(doc) {
  return JSON.parse(doc.getControls());
}

function firstTable(doc) {
  const t = controls(doc).find(x => x.ctrlId === 'tbl' && x.list === 0 && x.para === 0);
  if (!t) throw new Error('front-matter table not found');
  return { para: t.para, control: t.controlIndex };
}

function clearCell(doc, p, c, cell) {
  const count = doc.getCellParagraphCount(0, p, c, cell);
  if (!count) return;
  const end = count - 1;
  const length = doc.getCellParagraphLength(0, p, c, cell, end);
  ok(doc.deleteRangeInCell(0, p, c, cell, 0, 0, end, length), `clear cell ${cell}`);
}

function setCellParagraph(doc, p, c, cell, para, text, charProps, paraProps) {
  if (text) ok(doc.insertTextInCell(0, p, c, cell, para, 0, text), `insert cell ${cell}:${para}`);
  const len = doc.getCellParagraphLength(0, p, c, cell, para);
  if (len) ok(doc.applyCharFormatInCell(0, p, c, cell, para, 0, len, JSON.stringify(charProps)), `char cell ${cell}:${para}`);
  ok(doc.applyParaFormatInCell(0, p, c, cell, para, JSON.stringify(paraProps)), `para cell ${cell}:${para}`);
}

function rewriteFront(doc, anonymous) {
  const { para, control } = firstTable(doc);
  for (let cell = 0; cell < 7; cell += 1) clearCell(doc, para, control, cell);
  const author = anonymous ? '' : `${data.author} (${data.englishAuthor})`;
  const affiliation = anonymous ? '' : `${data.affiliation} · ${data.email} · 단독저자/교신저자`;
  setCellParagraph(doc, para, control, 0, 0, data.title, titleChar, { alignment: 'center', lineSpacing: 160, indent: 0, keepWithNext: true });
  setCellParagraph(doc, para, control, 1, 0, ' ', { ...frontBodyChar, fontSize: 600, textColor: '#ffffff' }, { alignment: 'center', lineSpacing: 100, indent: 0 });
  setCellParagraph(doc, para, control, 2, 0, data.englishTitle, { ...titleChar, fontSize: 1200 }, { alignment: 'center', lineSpacing: 130, indent: 0, keepWithNext: true });
  setCellParagraph(doc, para, control, 3, 0, author, authorChar, { alignment: 'center', lineSpacing: 160, indent: 0, keepWithNext: true });
  setCellParagraph(doc, para, control, 4, 0, `초록\n${data.abstract}\n\n핵심어: ${data.keywords}`, frontBodyChar, { alignment: 'justify', lineSpacing: 160, indent: 13.3 });
  setCellParagraph(doc, para, control, 5, 0, `Abstract\n${data.englishAbstract}\n\nKeywords: ${data.englishKeywords}`, { ...frontBodyChar, fontFamily: 'Palatino Linotype' }, { alignment: 'justify', lineSpacing: 160, indent: 13.3 });
  setCellParagraph(doc, para, control, 6, 0, affiliation, noteChar, { alignment: 'left', lineSpacing: 130, indent: 0 });
  ok(doc.setTableProperties(0, para, control, JSON.stringify({
    tableWidth: 51491, pageBreak: 2, repeatHeader: false, treatAsChar: true,
    paddingLeft: 141, paddingRight: 141, paddingTop: 141, paddingBottom: 141,
    borderLeft: BORDER_NONE, borderRight: BORDER_NONE, borderTop: BORDER_NONE, borderBottom: BORDER_NONE
  })), 'front table props');
  for (let cell = 0; cell < 7; cell += 1) {
    ok(doc.setCellProperties(0, para, control, cell, JSON.stringify({
      paddingLeft: 141, paddingRight: 141, paddingTop: 141, paddingBottom: 141,
      borderLeft: BORDER_NONE, borderRight: BORDER_NONE, borderTop: BORDER_NONE, borderBottom: BORDER_NONE,
      fillType: 'none', verticalAlign: 0
    })), `front cell props ${cell}`);
  }
}

function clearBody(doc) {
  const count = doc.getParagraphCount(0);
  // Keep paragraph 1 because the journal template stores its two-column definition there.
  for (let i = count - 1; i >= 2; i -= 1) ok(doc.deleteParagraph(0, i), `delete paragraph ${i}`);
}

function appendParagraph(doc, state, text, charProps=bodyChar, paraProps=pBody) {
  const idx = state.next;
  if (idx >= doc.getParagraphCount(0)) ok(doc.insertParagraph(0, idx - 1), `insert paragraph ${idx}`);
  if (text) ok(doc.insertText(0, idx, 0, text), `insert text ${idx}`);
  const len = doc.getParagraphLength(0, idx);
  if (len) ok(doc.applyCharFormat(0, idx, 0, len, JSON.stringify(charProps)), `char ${idx}`);
  ok(doc.applyParaFormat(0, idx, JSON.stringify(paraProps)), `para ${idx}`);
  state.next += 1;
  return idx;
}

function appendTable(doc, state, table) {
  appendParagraph(doc, state, table.title, tableTitleChar, { alignment: 'left', lineSpacing: 130, indent: 0, spacingBefore: 250, spacingAfter: 100, keepWithNext: true });
  const idx = state.next;
  if (idx >= doc.getParagraphCount(0)) ok(doc.insertParagraph(0, idx - 1), `insert table paragraph ${idx}`);
  const cols = table.headers.length;
  const usable = 23600;
  const weights = cols === 3 ? [0.18,0.34,0.48] : [0.14,0.17,0.20,0.245,0.245];
  const widths = weights.map(x => Math.floor(usable*x));
  widths[widths.length - 1] += usable - widths.reduce((a,b) => a+b,0);
  const made = ok(doc.createTableEx(JSON.stringify({ sectionIdx: 0, paraIdx: idx, charOffset: 0, rowCount: table.rows.length + 1, colCount: cols, treatAsChar: true, colWidths: widths })), `create table ${idx}`);
  const control = made.controlIdx;
  const all = [table.headers, ...table.rows];
  for (let r = 0; r < all.length; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const cell = r * cols + c;
      const value = String(all[r][c]);
      ok(doc.insertTextInCell(0, idx, control, cell, 0, 0, value), `table text ${r}:${c}`);
      const len = doc.getCellParagraphLength(0, idx, control, cell, 0);
      ok(doc.applyCharFormatInCell(0, idx, control, cell, 0, 0, len, JSON.stringify(r === 0 ? tableHeadChar : tableChar)), `table char ${r}:${c}`);
      ok(doc.applyParaFormatInCell(0, idx, control, cell, 0, JSON.stringify({ alignment: c === 0 ? 'center' : (r === 0 ? 'center' : 'left'), lineSpacing: 100, indent: 0 })), `table para ${r}:${c}`);
      const top = r === 0 ? BORDER_THICK : BORDER_NONE;
      const bottom = r === 0 ? BORDER_THIN : (r === all.length - 1 ? BORDER_THICK : BORDER_NONE);
      ok(doc.setCellProperties(0, idx, control, cell, JSON.stringify({
        paddingLeft: 100, paddingRight: 100, paddingTop: 70, paddingBottom: 70,
        borderLeft: BORDER_NONE, borderRight: BORDER_NONE, borderTop: top, borderBottom: bottom,
        fillType: 'none', verticalAlign: 1, isHeader: r === 0
      })), `table cell ${r}:${c}`);
    }
  }
  ok(doc.setTableProperties(0, idx, control, JSON.stringify({
    tableWidth: usable, pageBreak: 2, repeatHeader: true, treatAsChar: true,
    paddingLeft: 0, paddingRight: 0, paddingTop: 0, paddingBottom: 0,
    borderLeft: BORDER_NONE, borderRight: BORDER_NONE, borderTop: BORDER_NONE, borderBottom: BORDER_NONE
  })), `table props ${idx}`);
  state.next += 1;
  appendParagraph(doc, state, table.note, noteChar, { alignment: 'justify', lineSpacing: 130, indent: 0, spacingAfter: 200 });
}

function appendSection(doc, state, section) {
  appendParagraph(doc, state, section.heading, majorChar, pMajor);
  for (const p of section.paragraphs || []) appendParagraph(doc, state, p);
  for (const sub of section.subsections || []) {
    appendParagraph(doc, state, sub.heading, minorChar, pMinor);
    for (const p of sub.paragraphs || []) appendParagraph(doc, state, p);
    if (sub.table) appendTable(doc, state, sub.table);
    for (const p of sub.afterTable || []) appendParagraph(doc, state, p);
  }
}

function build(anonymous, outputName) {
  const doc = new HwpDocument(new Uint8Array(fs.readFileSync(templatePath)));
  clearBody(doc);
  rewriteFront(doc, anonymous);
  ok(doc.setColumnDef(0, 2, 0, 1, 2268), 'set body columns');
  const state = { next: 2 };
  appendParagraph(doc, state, anonymous ? '편집위원회 참고: 익명 심사용 원고' : data.technicalNote, noteChar, { alignment: 'justify', lineSpacing: 130, indent: 0, spacingAfter: 600 });
  for (const section of data.sections) appendSection(doc, state, section);
  appendParagraph(doc, state, '데이터와 코드 가용성', minorChar, pMinor);
  appendParagraph(doc, state, data.availability);
  const refHeading = appendParagraph(doc, state, '참고문헌', majorChar, { ...pMajor, pageBreakBefore: true });
  ok(doc.insertPageBreak(0, refHeading, 0), 'reference page break');
  for (const ref of data.references) appendParagraph(doc, state, ref, refChar, { alignment: 'justify', lineSpacing: 160, indent: -33.3, marginLeft: 33.3, spacingAfter: 130 });
  const report = doc.exportHwpWithReport();
  const loss = JSON.parse(report.contentLoss());
  if (loss.totalLosses > 0) throw new Error(`HWP export content loss: ${JSON.stringify(loss)}`);
  const bytes = report.takeBytes();
  const out = path.join(here, outputName);
  fs.writeFileSync(out, bytes);
  const reopened = new HwpDocument(bytes);
  const verify = JSON.parse(reopened.exportHwpVerify());
  if (verify.ok === false) throw new Error(`HWP verify failed: ${JSON.stringify(verify)}`);
  return { out: path.relative(root, out), bytes: bytes.length, pages: reopened.pageCount(), verify, loss };
}

const outputs = [
  build(false, 'kra-cultural-industry-submission-author.hwp'),
  build(true, 'kra-cultural-industry-submission-blind.hwp')
];
fs.writeFileSync(path.join(here, 'hwp-build-report.json'), JSON.stringify({ generatedAt: new Date().toISOString(), outputs }, null, 2));
console.log(JSON.stringify(outputs, null, 2));
