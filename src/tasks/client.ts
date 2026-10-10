/** The side that asks: a task call as a promise, or as a stream. */

import { CancelledError, TaskError, deserializeError } from "../errors.js";
import type { TaskMessage, Transport } from "./protocol.js";

export interface CallOptions {
  onProgress?: (value: unknown) => void;
  onChunk?: (chunk: unknown) => void;
  signal?: AbortSignal;
  transfer?: Transferable[];
}

interface Pending {
  resolve(value: unknown): void;
  reject(error: unknown): void;
  onProgress?: ((value: unknown) => void) | undefined;
  onChunk?: ((chunk: unknown) => void) | undefined;
}

export class TaskClient {
  #nextId = 1;
  readonly #pending = new Map<number, Pending>();
  readonly #stop: () => void;
  #tasks: string[] | null = null;
  readonly #ready: Promise<string[]>;
  #resolveReady!: (tasks: string[]) => void;

  constructor(readonly transport: Transport) {
    this.#ready = new Promise((resolve) => (this.#resolveReady = resolve));
    this.#stop = transport.listen((message) => this.#receive(message));

    transport.send({ kind: "describe" });
  }

  /** The tasks the host announced. Resolves once it has said hello. */
  ready(): Promise<string[]> {
    return this.#ready;
  }

  get tasks(): string[] | null {
    return this.#tasks;
  }

  #receive(message: TaskMessage): void {
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
        pending.resolve(message.result);
        return;
      case "error":
        this.#pending.delete(message.id);
        pending.reject(deserializeError(message.error));
    }
  }

  /** Call a task by name and await its result. */
  call<R = unknown>(task: string, payload: unknown, options: CallOptions = {}): Promise<R> {
    const id = this.#nextId++;
    return new Promise<R>((resolve, reject) => {
      if (options.signal?.aborted) {
        reject(new CancelledError(`'${task}' was cancelled before it started`));
        return;
      }
      this.#pending.set(id, {
        resolve: resolve as (value: unknown) => void,
        reject,
        onProgress: options.onProgress,
        onChunk: options.onChunk,
      });
      options.signal?.addEventListener(
        "abort",
        () => {
          if (this.#pending.has(id)) this.transport.send({ kind: "cancel", id });
        },
        { once: true },
      );
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
    for (const pending of this.#pending.values()) pending.reject(new TaskError(reason));
    this.#pending.clear();
    this.#stop();
    this.transport.close?.();
  }
}
