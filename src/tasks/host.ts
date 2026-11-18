/** The side that runs tasks: a registry, and a loop serving it on a transport. */

import { CancelledError, TaskError, serializeError } from "../errors.js";
import { readMessage, type TaskHandler, type Transport } from "./protocol.js";
import { DEFAULT_LIMITS, checkRemoteLocations, type LocationPolicy, type Trust } from "./trust.js";

export interface ServeOptions extends LocationPolicy {
  /**
   * How far the peer is trusted; by default what the transport says (a port: `local`, a socket:
   * `remote`). A `remote` peer's payloads may not name where code or models load from —
   * `wasmPaths`, `endpoint`, `resolverBase`, `hub`… — unless `allowedOrigins` / `allowedRepos` list
   * it: the page fixes those when it registers its handlers.
   */
  trust?: Trust;
  /**
   * Calls running at once on this transport; more are refused. Default 4 for a remote peer, 64 for
   * a local one.
   */
  maxConcurrentCalls?: number;
  /** The largest message a decoding transport accepts. Default 16 MiB. */
  maxMessageBytes?: number;
  /** The deepest nesting a message may have. Default 32. */
  maxDepth?: number;
}

export class TaskHost {
  readonly #tasks = new Map<string, TaskHandler>();

  /** Register a task by name; a library adds its own (`transcribe`, `embed`, `chat`…). */
  register<P, R>(name: string, handler: TaskHandler<P, R>): this {
    this.#tasks.set(name, handler as TaskHandler);
    return this;
  }

  get tasks(): string[] {
    return [...this.#tasks.keys()];
  }

  /** Serve on a transport until it closes; returns the function that stops. */
  serve(transport: Transport, options: ServeOptions = {}): () => void {
    const trust: Trust = options.trust ?? transport.peer ?? "local";
    const maxCalls = options.maxConcurrentCalls ?? (trust === "remote" ? 4 : 64);
    const maxDepth = options.maxDepth ?? DEFAULT_LIMITS.maxDepth;
    const policy: LocationPolicy = { ...(options.allowedOrigins ? { allowedOrigins: options.allowedOrigins } : {}), ...(options.allowedRepos ? { allowedRepos: options.allowedRepos } : {}) };
    const running = new Map<number, AbortController>();
    const fail = (id: number, error: unknown) => transport.send({ kind: "error", id, error: serializeError(error) });

    const stopListening = transport.listen(
      (raw) => {
        const read = readMessage(raw);
        if ("error" in read) {
          if (read.id !== null && !running.has(read.id)) fail(read.id, read.error);
          return;
        }
        const message = read.message;

        if (message.kind === "describe") {
          transport.send({ kind: "hello", tasks: this.tasks });
          return;
        }
        if (message.kind === "cancel") {
          running.get(message.id)?.abort(new CancelledError("cancelled by the caller"));
          return;
        }
        if (message.kind !== "call") return;

        if (running.has(message.id)) return void fail(message.id, new TaskError(`call ${message.id} is already running`));
        if (running.size >= maxCalls) {
          return void fail(message.id, new TaskError(`too many calls at once (${maxCalls}): wait for one to finish`, { details: { maxConcurrentCalls: maxCalls } }));
        }
        if (trust === "remote") {
          try {
            checkRemoteLocations(message.payload, policy, maxDepth);
          } catch (error) {
            return void fail(message.id, error);
          }
        }
        void this.#run(transport, running, trust, message.id, message.task, message.payload);
      },
      {
        limits: { maxDepth, ...(options.maxMessageBytes !== undefined ? { maxMessageBytes: options.maxMessageBytes } : {}) },
        onInvalid: (error, id) => {
          if (id !== null && !running.has(id)) fail(id, error);
        },
      },
    );
    const stop = () => {
      for (const controller of running.values()) controller.abort(new CancelledError("the host stopped serving"));
      running.clear();
      stopListening();
      stopClose?.();
    };

    const stopClose = transport.onClose?.(() => stop());
    transport.send({ kind: "hello", tasks: this.tasks });
    return stop;
  }

  async #run(transport: Transport, running: Map<number, AbortController>, trust: Trust, id: number, task: string, payload: unknown): Promise<void> {
    const handler = this.#tasks.get(task);
    if (!handler) {
      transport.send({
        kind: "error",
        id,
        error: serializeError(new TaskError(`no task named '${task}' here. Registered: ${this.tasks.join(", ") || "none"}.`, { details: { task, registered: this.tasks } })),
      });
      return;
    }
    const controller = new AbortController();
    running.set(id, controller);
    const transfer = new Set<ArrayBuffer>();
    try {
      const result = await handler(payload, {
        progress: (value) => transport.send({ kind: "progress", id, payload: value }),
        emit: (chunk) => transport.send({ kind: "chunk", id, payload: chunk }),
        transfer: (...buffers) => {
          for (const buffer of buffers) transfer.add(buffer);
        },
        signal: controller.signal,
        trust,
      });

      transport.send({ kind: "result", id, result }, [...transfer]);
    } catch (error) {
      const cause = controller.signal.aborted && !(error instanceof Error) ? controller.signal.reason : error;
      transport.send({ kind: "error", id, error: serializeError(cause) });
    } finally {
      running.delete(id);
    }
  }
}
