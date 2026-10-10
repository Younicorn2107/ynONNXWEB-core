/** The worker side: the page's task host, served on the worker's own scope. */

import { TaskHost } from "../tasks/host.js";
import type { TaskHandler } from "../tasks/protocol.js";
import { portTransport } from "../tasks/transports.js";

/** The host every task of this worker is registered on. */
export const workerHost = new TaskHost();

/** Register a task by name; a library registers its own (`transcribe`, `embed`…). */
export function registerTask<P, R>(name: string, task: TaskHandler<P, R>): void {
  workerHost.register(name, task);
}

export function registeredTasks(): string[] {
  return workerHost.tasks;
}

let serving = false;

/** Start serving on the worker's scope. Called by the bundled entry, once. */
export function serve(): void {
  if (serving) return;
  serving = true;
  workerHost.serve(portTransport(globalThis as unknown as Parameters<typeof portTransport>[0]));
}

if (
  typeof (globalThis as { importScripts?: unknown }).importScripts === "function" ||
  (globalThis as { constructor?: { name?: string } }).constructor?.name === "DedicatedWorkerGlobalScope"
) {
  serve();
}
