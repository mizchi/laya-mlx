/**
 * A model-free stand-in for the real Laya agent. It answers questions with a
 * cheap, deterministic heuristic instead of running inference, so the policy
 * and shield logic can be exercised without a checkpoint.
 */
import type {
  Answer,
  ChoiceAnswer,
  ChoiceQuestion,
  NoulAnswer,
  NoulQuestion,
  PredictResult,
  Question,
  ScoreAnswer,
  ScoreQuestion,
  State,
} from "@laya-mlx/web";

export interface StubAgentOptions {
  probability?: (label: string, description: string) => number;
}

function optionEntries(criteria: ChoiceQuestion["criteria"]): [string, string][] {
  if (Array.isArray(criteria)) {
    return criteria.map((label) => [label, ""]);
  }
  return Object.entries(criteria).map(([label, description]) => [label, String(description)]);
}

function defaultProbability(_label: string, description: string): number {
  if (description.includes("Best")) return 0.7;
  if (description.startsWith("Safe")) return 0.2;
  return 0.05;
}

function answerChoice(
  question: ChoiceQuestion,
  probability: (label: string, description: string) => number,
): ChoiceAnswer {
  const entries = optionEntries(question.criteria);
  const raw = entries.map(
    ([label, description]) => [label, probability(label, description)] as const,
  );
  const total = raw.reduce((sum, [, value]) => sum + value, 0);
  const probabilities: Record<string, number> = {};
  let choice = raw[0]?.[0] ?? "";
  let best = -Infinity;
  for (const [label, value] of raw) {
    probabilities[label] = value / total;
    if (value > best) {
      best = value;
      choice = label;
    }
  }
  return {
    type: "choice",
    confidence: 0.5,
    action: { act_probability: 1 },
    choice,
    probabilities,
  };
}

function answerNoul(_question: NoulQuestion, state: State): NoulAnswer {
  const noul = /yes/.test(String(state)) ? 0.9 : 0.4;
  return {
    type: "noul",
    confidence: noul,
    action: { act_probability: 1 },
    noul,
  };
}

function answerScore(_question: ScoreQuestion): ScoreAnswer {
  return {
    type: "score",
    confidence: 0.5,
    action: { act_probability: 1 },
    score: 0,
    legend: {},
    probabilities: { "0": 1 },
  };
}

export class StubAgent {
  private readonly probability: (label: string, description: string) => number;

  constructor(options: StubAgentOptions = {}) {
    this.probability = options.probability ?? defaultProbability;
  }

  async predict(state: State, questions: Record<string, Question>): Promise<PredictResult> {
    const answers: Record<string, Answer> = {};
    for (const [key, question] of Object.entries(questions)) {
      if (question.type === "choice") {
        answers[key] = answerChoice(question, this.probability);
      } else if (question.type === "noul") {
        answers[key] = answerNoul(question, state);
      } else {
        answers[key] = answerScore(question);
      }
    }
    const serializedLength = JSON.stringify(state).length + JSON.stringify(questions).length;
    return {
      model: "laya-rl-agent",
      answers,
      usage: { input_tokens: Math.ceil(serializedLength / 4), output_tokens: 0 },
    };
  }
}
