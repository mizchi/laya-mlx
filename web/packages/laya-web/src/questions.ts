import { pyJson } from "./pyjson.ts";
import type { Criterion, InternalQuestion, Json, Question, QuestionType } from "./types.ts";
import { QTYPES } from "./types.ts";

const isPlainObject = (v: unknown): v is Record<string, Json> =>
  typeof v === "object" && v !== null && !Array.isArray(v);

/** Port of Python `Agent._to_internal`, including its error messages. */
export function toInternal(question: Question): InternalQuestion {
  if (!isPlainObject(question)) throw new Error("Each question must be a dictionary");
  const kind = question.type as QuestionType;
  if (!(kind in QTYPES)) {
    throw new Error(
      `Unknown question type ${JSON.stringify(kind)}; expected choice, score, or noul`,
    );
  }
  if (!("instructions" in question)) throw new Error("Question is missing instructions");
  const raw = question.instructions;
  const ins = typeof raw === "string" ? raw : pyJson(raw as Json);
  let criteria: unknown = (question as { criteria?: unknown }).criteria;
  if (kind === "choice") {
    if (Array.isArray(criteria)) {
      if (!criteria.every((c) => typeof c === "string"))
        throw new Error("Choice labels must be strings");
      if (new Set(criteria).size !== criteria.length)
        throw new Error("Choice labels must be unique");
      criteria = Object.fromEntries((criteria as string[]).map((c) => [c, null]));
    }
    if (!isPlainObject(criteria) || Object.keys(criteria).length === 0) {
      throw new Error("Choice criteria must be a nonempty dictionary or list");
    }
    return { t: "choice", ins, crit: criteria as Record<string, Criterion | null> };
  }
  if (kind === "score") {
    if (!Array.isArray(criteria) || criteria.length === 0) {
      throw new Error("Score criteria must be a nonempty list");
    }
    return { t: "score", ins, crit: criteria as Criterion[] };
  }
  if (criteria !== undefined && criteria !== null && !isPlainObject(criteria)) {
    throw new Error("Noul criteria must be a dictionary with false/true descriptions");
  }
  return {
    t: "noul",
    ins,
    crit: (criteria as { false?: Criterion; true?: Criterion } | null | undefined) ?? null,
  };
}

/** Python `render_criterion`: strings pass through, structured values become compact JSON. */
export function renderCriterion(value: Criterion): string {
  return typeof value === "string" ? value : pyJson(value, { ensureAscii: false });
}

const isBlank = (v: Criterion | null | undefined) => v === null || v === undefined || v === "";

/** Python `render_options`: option texts in label-index order; noul is always [false, true]. */
export function renderOptions(q: InternalQuestion): string[] {
  if (q.t === "choice") {
    return Object.entries(q.crit).map(([k, v]) =>
      isBlank(v) ? k : `${k}: ${renderCriterion(v!)}`,
    );
  }
  if (q.t === "score") return q.crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  const crit = q.crit ?? {};
  return [
    "false: " +
      (isBlank(crit.false) ? "no, the statement does not hold" : renderCriterion(crit.false!)),
    "true: " + (isBlank(crit.true) ? "yes, the statement holds" : renderCriterion(crit.true!)),
  ];
}
