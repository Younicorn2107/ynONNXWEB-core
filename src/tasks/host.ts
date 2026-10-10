/** The side that runs tasks: a registry, and a loop serving it on a transport. */

import { CancelledError, TaskError, serializeError } from "../errors.js";
import type { TaskHandler, Transport } from "./protocol.js";

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
  serve(transport: Transport): () => void {
    const running = new Map<number, AbortController>();
    const stop = transport.listen((message) => {
      if (message.kind === "describe") {
        transport.send({ kind: "hello", tasks: this.tasks });
        return;
      }
      if (message.kind === "cancel") {
        running.get(message.id)?.abort(new CancelledError("cancelled by the caller"));
        return;
      }
      if (message.kind === "call") void this.#run(transport, running, message.id, message.task, message.payload);
    });
    transport.send({ kind: "hello", tasks: this.tasks });
    return () => {
      for (const controller of running.values()) controller.abort(new CancelledError("the host stopped serving"));
      running.clear();
      stop();
    };
  }

  async #run(
    transport: Transport,
    running: Map<number, AbortController>,
    id: number,
    task: string,
    payload: unknown,
  ): Promise<void> {
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
    try {
      const result = await handler(payload, {
        progress: (value) => transport.send({ kind: "progress", id, payload: value }),
        emit: (chunk) => transport.send({ kind: "chunk", id, payload: chunk }),
        signal: controller.signal,
      });
      transport.send({ kind: "result", id, result }, transferablesOf(result));
    } catch (error) {
      const cause = controller.signal.aborted && !(error instanceof Error) ? controller.signal.reason : error;
      transport.send({ kind: "error", id, error: serializeError(cause) });
    } finally {
      running.delete(id);
    }
  }
}

/** The buffers of a result's typed arrays, handed over rather than copied. */
export function transferablesOf(value: unknown, found: Set<ArrayBuffer> = new Set(), depth = 0): Transferable[] {
  if (depth > 6 || value === null || typeof value !== "object") return [...found];
  if (ArrayBuffer.isView(value)) {
    if (value.buffer instanceof ArrayBuffer && value.byteLength === value.buffer.byteLength) found.add(value.buffer);
  } else if (value instanceof ArrayBuffer) {
    found.add(value);
  } else if (Array.isArray(value)) {
    for (const item of value.slice(0, 64)) transferablesOf(item, found, depth + 1);
  } else {
    for (const item of Object.values(value)) transferablesOf(item, found, depth + 1);
  }
  return [...found];
}
