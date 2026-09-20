/**
 * Generic first-max-wins argmax, shared by the Snake and Chess policies to turn
 * a probability distribution into a single deterministic proposal: the first
 * item to reach the highest score wins ties, matching the argmax helpers both
 * policies used to keep locally.
 */
export function argmax<T>(items: readonly T[], score: (item: T) => number): T {
  if (items.length === 0) {
    throw new Error("argmax: empty input");
  }
  let best: T = items[0]!;
  let bestScore = score(best);
  for (const item of items) {
    const value = score(item);
    if (value > bestScore) {
      bestScore = value;
      best = item;
    }
  }
  return best;
}
