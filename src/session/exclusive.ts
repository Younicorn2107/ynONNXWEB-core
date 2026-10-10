/** One onnxruntime call at a time: creating, running and releasing sessions. */

let queue: Promise<unknown> = Promise.resolve();

export function exclusive<T>(work: () => Promise<T>): Promise<T> {
  const next = queue.then(work, work);
  queue = next.catch(() => undefined);
  return next;
}
