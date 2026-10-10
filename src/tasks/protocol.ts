/** Tasks a page runs on behalf of someone else, over any transport. */

import type { SerializedError } from "../errors.js";

export type TaskMessage =
  | { kind: "describe" }
  | { kind: "call"; id: number; task: string; payload: unknown }
  | { kind: "cancel"; id: number }
  | { kind: "hello"; tasks: string[] }
  | { kind: "progress"; id: number; payload: unknown }
  | { kind: "chunk"; id: number; payload: unknown }
  | { kind: "result"; id: number; result: unknown }
  | { kind: "error"; id: number; error: SerializedError };

/** One end of a channel. */
export interface Transport {
  send(message: TaskMessage, transfer?: Transferable[]): void;
  /** Subscribe; returns the unsubscribe function. */
  listen(handler: (message: TaskMessage) => void): () => void;
  close?(): void;
}

export interface TaskContext {
  /** Report how far along the task is; the caller sees it as it happens. */
  progress(value: unknown): void;
  /** Send part of the result before the end (tokens, segments). */
  emit(chunk: unknown): void;
  /** Aborted when the caller cancels or goes away. */
  signal: AbortSignal;
}

export type TaskHandler<P = unknown, R = unknown> = (payload: P, context: TaskContext) => Promise<R>;
