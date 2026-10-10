/** The page side: talk to a worker as if it were a promise-returning object. */

import { ConfigurationError } from "../errors.js";
import type { Logger } from "../logger.js";
import { TaskClient, type CallOptions } from "../tasks/client.js";
import { portTransport } from "../tasks/transports.js";

export interface WorkerClientOptions {
  /** URL of the bundled worker script. */
  url: string;
  logger?: Logger;
  name?: string;
}

/** Load a worker, falling back to a Blob URL when the script is cross-origin. */
export async function createWorker(url: string, name?: string): Promise<Worker> {
  try {
    return new Worker(url, { type: "module", ...(name ? { name } : {}) });
  } catch {
    const response = await fetch(url);
    if (!response.ok) {
      throw new ConfigurationError(
        `the worker script at ${url} could not be loaded (HTTP ${response.status}). ` +
          `When static files are served from another origin, the script is fetched ` +
          `and run from a Blob URL, so it must also be readable by CORS.`,
      );
    }
    const blob = new Blob([await response.text()], { type: "text/javascript" });
    const blobUrl = URL.createObjectURL(blob);
    try {
      return new Worker(blobUrl, { type: "module", ...(name ? { name } : {}) });
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
  }
}

export class WorkerClient {
  #client: TaskClient | null = null;
  #worker: Worker | null = null;

  constructor(private readonly options: WorkerClientOptions) {}

  async #connect(): Promise<TaskClient> {
    if (this.#client) return this.#client;
    const worker = await createWorker(this.options.url, this.options.name);
    const client = new TaskClient(portTransport(worker as unknown as Parameters<typeof portTransport>[0]));
    worker.addEventListener("error", (event) => {
      this.options.logger?.error(`the inference worker failed: ${event.message}`);
      client.close(`the inference worker failed: ${event.message}`);
      this.#client = null;
      this.#worker = null;
    });
    this.#worker = worker;
    this.#client = client;
    return client;
  }

  /** Call a task by name and await its result. */
  async call<R = unknown>(task: string, payload: unknown, options: CallOptions = {}): Promise<R> {
    return (await this.#connect()).call<R>(task, payload, options);
  }

  /** Call a task and read what it emits as it comes. */
  async *stream<C = unknown, R = unknown>(task: string, payload: unknown, options: Omit<CallOptions, "onChunk"> = {}): AsyncGenerator<C, R, undefined> {
    return yield* (await this.#connect()).stream<C, R>(task, payload, options);
  }

  /** Stop the worker. Pending calls reject rather than hang for ever. */
  terminate(): void {
    this.#client?.close("the inference worker was terminated");
    this.#client = null;
    this.#worker?.terminate();
    this.#worker = null;
  }
}
