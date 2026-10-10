/** Prompt plumbing every library would otherwise duplicate. */

/** The lengths a carried context is rounded down to. */
export const CONTEXT_STEPS: readonly number[] = [224, 128, 64, 32, 0];

/**
 * Shrink a context to the next step down the ladder, keeping its END: what conditions the next
 * window is the last words spoken.
 */
export function quantizeContext(context: readonly number[], steps: readonly number[] = CONTEXT_STEPS): number[] {
  if (context.length === 0) return [];
  const ladder = [...new Set(steps.filter((s) => s >= 0).map(Math.trunc))].sort((a, b) => b - a);
  if (ladder.length === 0) return [...context];
  for (const step of ladder) {
    if (context.length >= step) return step > 0 ? context.slice(-step) : [];
  }
  const last = ladder[ladder.length - 1]!;
  return last > 0 ? context.slice(-last) : [];
}

/** `prefix + context (truncated from the left) + start`. */
export function assemblePrompt(
  prefix: readonly number[],
  context: readonly number[] | null,
  start: readonly number[],
  maxContext = 224,
): number[] {
  if (!context || context.length === 0 || maxContext <= 0) return [...start];
  const kept = context.slice(-maxContext);
  return kept.length > 0 ? [...prefix, ...kept, ...start] : [...start];
}

/** Prompt indices grouped by exact length; each group batches as is. */
export function bucketByLength(prompts: readonly (readonly number[])[]): Map<number, number[]> {
  const buckets = new Map<number, number[]>();
  prompts.forEach((prompt, index) => {
    const group = buckets.get(prompt.length) ?? [];
    group.push(index);
    buckets.set(prompt.length, group);
  });
  return new Map([...buckets].sort(([a], [b]) => a - b));
}
