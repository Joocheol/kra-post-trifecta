import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";

const raw = process.env.CLAUDE_REVIEW;
const outputPath = process.env.GITHUB_OUTPUT;

if (!raw) throw new Error("CLAUDE_REVIEW is empty.");
if (!outputPath) throw new Error("GITHUB_OUTPUT is unavailable.");
if (Buffer.byteLength(raw, "utf8") > 512 * 1024) {
  throw new Error("Claude review exceeds the 512 KiB handoff limit.");
}

let review;
try {
  review = JSON.parse(raw);
} catch (error) {
  throw new Error(`Claude review is not valid JSON: ${error.message}`);
}

const topKeys = ["author_questions", "findings", "needs_changes", "summary"];
const findingKeys = [
  "category",
  "evidence",
  "file",
  "id",
  "line",
  "recommendation",
  "requires_author_decision",
  "severity",
];

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

function assertExactKeys(value, expected, label) {
  assert(value && typeof value === "object" && !Array.isArray(value), `${label} must be an object.`);
  const actual = Object.keys(value).sort();
  assert(JSON.stringify(actual) === JSON.stringify(expected), `${label} has unexpected or missing keys.`);
}

function assertText(value, label, maxLength, allowEmpty = false) {
  assert(typeof value === "string", `${label} must be a string.`);
  assert(!/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/u.test(value), `${label} contains control characters.`);
  if (!allowEmpty) assert(value.trim().length > 0, `${label} must not be empty.`);
  assert(value.length <= maxLength, `${label} exceeds ${maxLength} characters.`);
}

assertExactKeys(review, topKeys, "review");
assert(typeof review.needs_changes === "boolean", "needs_changes must be boolean.");
assertText(review.summary, "summary", 6000, true);
assert(Array.isArray(review.findings) && review.findings.length <= 15, "findings must contain at most 15 items.");
assert(Array.isArray(review.author_questions) && review.author_questions.length <= 5, "author_questions must contain at most 5 items.");

const ids = new Set();
for (const [index, finding] of review.findings.entries()) {
  const label = `findings[${index}]`;
  assertExactKeys(finding, findingKeys, label);
  assertText(finding.id, `${label}.id`, 80);
  assert(/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(finding.id), `${label}.id has an unsafe format.`);
  assert(!ids.has(finding.id), `${label}.id is duplicated.`);
  ids.add(finding.id);
  assert(["high", "medium", "low"].includes(finding.severity), `${label}.severity is invalid.`);
  assertText(finding.category, `${label}.category`, 120);
  assertText(finding.file, `${label}.file`, 500);
  const normalized = path.posix.normalize(finding.file.replaceAll("\\", "/"));
  assert(!path.posix.isAbsolute(normalized), `${label}.file must be repository-relative.`);
  assert(normalized !== ".." && !normalized.startsWith("../"), `${label}.file escapes the repository.`);
  assert(Number.isInteger(finding.line) && finding.line >= 0, `${label}.line must be nonnegative.`);
  assertText(finding.evidence, `${label}.evidence`, 4000);
  assertText(finding.recommendation, `${label}.recommendation`, 4000);
  assert(typeof finding.requires_author_decision === "boolean", `${label}.requires_author_decision must be boolean.`);
}

for (const [index, question] of review.author_questions.entries()) {
  assertText(question, `author_questions[${index}]`, 2000);
}

// Treat the structured findings as the source of truth for the derived verdict.
// Claude may occasionally emit an internally inconsistent boolean (for example,
// needs_changes=false while also returning a concrete non-author-decision fix).
// That is a serialization inconsistency, not a reason to discard an otherwise
// valid academic review. Normalize only this derived field; never alter findings,
// severities, evidence, recommendations, or author-decision flags.
const computedNeedsChanges = review.findings.some(
  (finding) => !finding.requires_author_decision,
);
if (review.needs_changes !== computedNeedsChanges) {
  console.warn(
    `::warning::Claude review needs_changes normalized from ${review.needs_changes} ` +
      `to ${computedNeedsChanges}; findings remain unchanged.`,
  );
}
review.needs_changes = computedNeedsChanges;

const canonical = JSON.stringify(review);
const delimiter = `CLAUDE_REVIEW_${crypto.randomUUID()}`;
fs.appendFileSync(
  outputPath,
  [
    `review<<${delimiter}`,
    canonical,
    delimiter,
    `summary<<${delimiter}`,
    review.summary,
    delimiter,
    `needs_changes=${computedNeedsChanges}`,
    "",
  ].join("\n"),
);
