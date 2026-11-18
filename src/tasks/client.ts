/** The side that asks: a task call as a promise, or as a stream. */

import { CancelledError, TaskError, deserializeError } from "../errors.js";
import { readMessage, type TaskMessage, type Transport } from "./protocol.js";
import type { WireLimits } from "./trust.js";

export interface CallOptions {
  onProgress?: (value: unknown) => void;
  onChunk?: (chunk: unknown) => void;
  signal?: AbortSignal;
  transfer?: Transferable[];
}

export interface TaskClientOptions {
  /** Decoding limits for what the host sends back (a transport's own defaults otherwise). */
  limits?: Partial<WireLimits>;
}

interface Pending {
  resolve(value: unknown): void;
  reject(error: unknown): void;
  onProgress?: ((value: unknown) => void) | undefined;
  onChunk?: ((chunk: unknown) => void) | undefined;
  /** Removes the caller's abort listener: a long-lived signal must not collect one per call. */
  detach(): void;
}

export class TaskClient {
  #nextId = 1;
  readonly #pending = new Map<number, Pending>();
  readonly #stop: () => void;
  readonly #stopClose: (() => void) | undefined;
  #closed: string | null = null;
  #tasks: string[] | null = null;
  readonly #ready: Promise<string[]>;
  #resolveReady!: (tasks: string[]) => void;
  #rejectReady!: (error: unknown) => void;

  constructor(
    readonly transport: Transport,
    options: TaskClientOptions = {},
  ) {
    this.#ready = new Promise((resolve, reject) => {
      this.#resolveReady = resolve;
      this.#rejectReady = reject;
    });

    this.#ready.catch(() => undefined);
    this.#stop = transport.listen((message) => this.#receive(message), {
      ...(options.limits ? { limits: options.limits } : {}),

      onInvalid: (error, id) => this.#settle(id, error),
    });

    this.#stopClose = transport.onClose?.((reason) => this.close(reason));

    transport.send({ kind: "describe" });
  }

  /** The tasks the host announced. Resolves once it has said hello. */
  ready(): Promise<string[]> {
    return this.#ready;
  }

  get tasks(): string[] | null {
    return this.#tasks;
  }

  #settle(id: number | null, error: unknown): void {
    if (id === null) return;
    const pending = this.#pending.get(id);
    if (!pending) return;
    this.#pending.delete(id);
    pending.detach();
    pending.reject(error);
  }

  #receive(raw: unknown): void {
    const read = readMessage(raw);
    if ("error" in read) return this.#settle(read.id, read.error);
    const message: TaskMessage = read.message;
    if (message.kind === "hello") {
      this.#tasks = message.tasks;
      this.#resolveReady(message.tasks);
      return;
    }
    if (message.kind === "call" || message.kind === "cancel" || message.kind === "describe") return;
    const pending = this.#pending.get(message.id);
    if (!pending) return;
    switch (message.kind) {
      case "progress":
        pending.onProgress?.(message.payload);
        return;
      case "chunk":
        pending.onChunk?.(message.payload);
        return;
      case "result":
        this.#pending.delete(message.id);
        pending.detach();
        pending.resolve(message.result);
        return;
      case "error":
        this.#settle(message.id, deserializeError(message.error));
    }
  }

  /** Call a task by name and await its result. */
  call<R = unknown>(task: string, payload: unknown, options: CallOptions = {}): Promise<R> {
    const id = this.#nextId++;
    return new Promise<R>((resolve, reject) => {
      if (this.#closed !== null) {
        reject(new TaskError(this.#closed));
        return;
      }
      const signal = options.signal;
      if (signal?.aborted) {
        reject(new CancelledError(`'${task}' was cancelled before it started`));
        return;
      }
      const onAbort = () => {
        if (this.#pending.has(id)) this.transport.send({ kind: "cancel", id });
      };
      this.#pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        onProgress: options.onProgress,
        onChunk: options.onChunk,
        detach: () => signal?.removeEventListener("abort", onAbort),
      });
      signal?.addEventListener("abort", onAbort, { once: true });
      this.transport.send({ kind: "call", id, task, payload }, options.transfer);
    });
  }

  /**
   * Call a task and read what it emits as it comes; the iterator's return value is the task's
   * result.
   */
  async *stream<C = unknown, R = unknown>(
    task: string,
    payload: unknown,
    options: Omit<CallOptions, "onChunk"> = {},
  ): AsyncGenerator<C, R, undefined> {
    const queue: C[] = [];
    let wake: (() => void) | null = null;
    let finished = false;
    const done = this.call<R>(task, payload, {
      ...options,
      onChunk: (chunk) => {
        queue.push(chunk as C);
        wake?.();
      },
    }).finally(() => {
      finished = true;
      wake?.();
    });

    done.catch(() => undefined);
    for (;;) {
      while (queue.length > 0) yield queue.shift()!;
      if (finished) break;
      await new Promise<void>((resolve) => (wake = resolve));
      wake = null;
    }
    return await done;
  }

  /** Reject every pending call and stop listening. */
  close(reason = "the connection closed"): void {
    if (this.#closed !== null) return;
    this.#closed = reason;
    for (const pending of this.#pending.values()) {
      pending.detach();
      pending.reject(new TaskError(reason));
    }
    this.#pending.clear();
    this.#rejectReady(new TaskError(reason));
    this.#stop();
    this.#stopClose?.();
    this.transport.close?.();
  }
}
