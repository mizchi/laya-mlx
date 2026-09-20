import { argmax, loadAgent, softmax, tempBucket } from "@laya-mlx/web";
import type { AgentConfig, PredictResult, PreparedItem, Question, State } from "@laya-mlx/web";

import { MODEL_URL, ORT_WASM_PATHS } from "./model-url.ts";

interface FixtureCase {
  name: string;
  state: State;
  questions: Record<string, Question>;
  items: PreparedItem[];
  logits: number[][];
  result: PredictResult;
}
interface Fixture {
  config: AgentConfig;
  cases: FixtureCase[];
}

/** One question's browser-vs-Python comparison: does the top choice agree, and by how much do the two probability distributions differ at most. */
interface QuestionScore {
  argmaxAgrees: boolean;
  maxAbsError: number;
}

function scoreQuestion(expected: number[], actual: number[]): QuestionScore {
  return {
    argmaxAgrees: argmax(expected) === argmax(actual),
    maxAbsError: expected.reduce((worst, v, i) => Math.max(worst, Math.abs(v - actual[i]!)), 0),
  };
}

export interface ParityReport {
  provider: string;
  questions: number;
  argmax_agreements: number;
  probability_max_abs_error: number;
  public_result_equal: number;
  timings_ms: Record<string, number>;
}

const status = document.getElementById("status")!;
const report = document.getElementById("report")!;
const modelUrl = new URLSearchParams(location.search).get("model") ?? MODEL_URL;

async function main(): Promise<ParityReport> {
  const fixture: Fixture = await (
    await fetch(`${import.meta.env.BASE_URL}fixtures/parity-multilingual.json`)
  ).json();
  status.textContent = `loading ${modelUrl}`;
  const { agent, provider } = await loadAgent(modelUrl, {
    wasmPaths: ORT_WASM_PATHS,
    batchSize: 64,
    onProgress: (p) => {
      status.textContent = `${p.file}: ${(p.received / 1e6).toFixed(0)} / ${p.total ? (p.total / 1e6).toFixed(0) : "?"} MB`;
    },
  });
  status.textContent = `running on ${provider}`;
  const summary: ParityReport = {
    provider,
    questions: 0,
    argmax_agreements: 0,
    probability_max_abs_error: 0,
    public_result_equal: 0,
    timings_ms: {},
  };
  for (const c of fixture.cases) {
    const started = performance.now();
    const result = await agent.predict(c.state, c.questions);
    summary.timings_ms[c.name] = Math.round(performance.now() - started);
    const ids = Object.keys(c.questions);
    ids.forEach((qid, row) => {
      const k = c.items[row]!.markers.length;
      const qtype = c.items[row]!.qtype;
      const scale =
        fixture.config.temperature_by_options[tempBucket(qtype, k)] ??
        fixture.config.temperature[qtype]!;
      const expected = softmax(c.logits[row]!.slice(0, k).map((v) => v / Math.max(1e-3, scale)));
      const answer = result.answers[qid]!;
      const actual =
        answer.type === "noul"
          ? [1 - answer.noul, answer.noul]
          : Object.values(answer.probabilities);
      const { argmaxAgrees, maxAbsError } = scoreQuestion(expected, actual);
      summary.questions += 1;
      if (argmaxAgrees) summary.argmax_agreements += 1;
      summary.probability_max_abs_error = Math.max(summary.probability_max_abs_error, maxAbsError);
    });
    if (JSON.stringify(result) === JSON.stringify(c.result)) summary.public_result_equal += 1;
  }
  status.textContent = `done: ${summary.argmax_agreements}/${summary.questions} argmax on ${provider}`;
  report.textContent = JSON.stringify(summary, null, 2);
  return summary;
}

main()
  .then((summary) => console.log("[laya] RESULT " + JSON.stringify(summary)))
  .catch((error) => {
    status.textContent = `ERROR ${String(error)}`;
    console.error(error);
  });
