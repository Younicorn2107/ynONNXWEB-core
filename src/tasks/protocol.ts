/** Tasks a page runs on behalf of someone else, over any transport. */

import { TaskError, type SerializedError } from "../errors.js";
import type { Trust, WireLimits } from "./trust.js";

export type TaskMessage =
  | { kind: "describe" }
  | { kind: "call"; id: number; task: string; payload: unknown }
  | { kind: "cancel"; id: number }
  | { kind: "hello"; tasks: string[] }
  | { kind: "progress"; id: number; payload: unknown }
  | { kind: "chunk"; id: number; payload: unknown }
  | { kind: "result"; id: number; result: unknown }
  | { kind: "error"; id: number; error: SerializedError };

/**
 * What a transport hands over: a message, or why one could not be read (and its id, if that much
 * could).
 */
export interface ListenOptions {
  /**
   * Applied by transports that decode text (a WebSocket); a port's structured clone is the page's
   * own.
   */
  limits?: Partial<WireLimits>;
  /** A message that arrived but could not be decoded. */
  onInvalid?(error: TaskError, id: number | null): void;
}

/** One end of a channel. */
export interface Transport {
  send(message: TaskMessage, transfer?: Transferable[]): void;
  /**
   * Subscribe; returns the unsubscribe function. What reaches `handler` is only decoded, not
   * validated: the host and the client run `readMessage`.
   */
  listen(handler: (message: unknown) => void, options?: ListenOptions): () => void;
  /**
   * Called once when the channel ends on its own (a socket closed or failed); returns the
   * unsubscribe function.
   */
  onClose?(handler: (reason: string) => void): () => void;
  close?(): void;
  /**
   * Who is at the other end; a host serving this transport trusts it that much unless told
   * otherwise.
   */
  readonly peer?: Trust;
}

export interface TaskContext {
  /** Report how far along the task is; the caller sees it as it happens. */
  progress(value: unknown): void;
  /** Send part of the result before the end (tokens, segments). */
  emit(chunk: unknown): void;
  /**
   * Hand these buffers over with the result instead of copying them. Only for buffers this handler
   * will never touch again: a transferred buffer is detached on this side, so a cached or reused
   * output must not be marked.
   */
  transfer(...buffers: ArrayBuffer[]): void;
  /** Aborted when the caller cancels or goes away. */
  signal: AbortSignal;
  /** How far the caller is trusted: a handler serving `remote` peers checks what it is asked. */
  trust: Trust;
}

export type TaskHandler<P = unknown, R = unknown> = (payload: P, context: TaskContext) => Promise<R>;

const KINDS: ReadonlySet<string> = new Set(["describe", "call", "cancel", "hello", "progress", "chunk", "result", "error"]);
const MAX_TASK_NAME = 200;
const MAX_TASKS = 1000;
const MAX_ERROR_TEXT = 10_000;
const MAX_CAUSES = 8;

/** The id of a message, when it has one that can be trusted as an id. */
export function messageId(value: unknown): number | null {
  if (value === null || typeof value !== "object") return null;
  const id = (value as { id?: unknown }).id;
  return typeof id === "number" && Number.isSafeInteger(id) && id >= 0 ? id : null;
}

function text(value: unknown, fallback: string, max: number): string {
  return typeof value === "string" ? value.slice(0, max) : fallback;
}

/** An error from the far side, rebuilt from known fields only. */
function readError(value: unknown, depth = 0): SerializedError {
  const raw = value !== null && typeof value === "object" ? (value as Record<string, unknown>) : {};
  const type = text(raw["type"], text(raw["code"], "Error", MAX_TASK_NAME), MAX_TASK_NAME);
  const out: SerializedError = {
    type,
    code: text(raw["code"], type, MAX_TASK_NAME),
    message: text(raw["message"], value === undefined ? "the peer sent an error without a message" : "", MAX_ERROR_TEXT),
    retryable: raw["retryable"] === true,
  };
  const details = raw["details"];
  if (details !== null && typeof details === "object" && !Array.isArray(details)) out.details = details as Record<string, unknown>;
  if (raw["cause"] !== undefined && depth < MAX_CAUSES) out.cause = readError(raw["cause"], depth + 1);
  return out;
}

function invalid(why: string, value: unknown): { error: TaskError; id: number | null } {
  return { error: new TaskError(`invalid task message: ${why}`, { details: { reason: why } }), id: messageId(value) };
}

/**
 * Check a decoded message's envelope and return a fresh one holding the known fields only. A
 * payload is the task's to validate; the envelope is this module's.
 */
export function readMessage(value: unknown): { message: TaskMessage } | { error: TaskError; id: number | null } {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return invalid("not an object", value);
  const raw = value as Record<string, unknown>;
  const kind = raw["kind"];
  if (typeof kind !== "string" || !KINDS.has(kind)) return invalid(`unknown kind ${JSON.stringify(String(kind).slice(0, 40))}`, value);
  if (kind === "describe") return { message: { kind } };
  if (kind === "hello") {
    const tasks = raw["tasks"];
    if (!Array.isArray(tasks) || tasks.length > MAX_TASKS || !tasks.every((t) => typeof t === "string" && t.length <= MAX_TASK_NAME)) {
      return invalid("hello without a list of task names", value);
    }
    return { message: { kind, tasks: [...tasks] as string[] } };
  }
  const id = messageId(value);
  if (id === null) return invalid("the id is not a non-negative safe integer", value);
  switch (kind) {
    case "call": {
      const task = raw["task"];
      if (typeof task !== "string" || task.length === 0 || task.length > MAX_TASK_NAME) return invalid("a call without a task name", value);
      return { message: { kind, id, task, payload: raw["payload"] } };
    }
    case "cancel":
      return { message: { kind, id } };
    case "progress":
    case "chunk":
      return { message: { kind, id, payload: raw["payload"] } };
    case "result":
      return { message: { kind, id, result: raw["result"] } };
    default:
      return { message: { kind: "error", id, error: readError(raw["error"]) } };
  }
}
