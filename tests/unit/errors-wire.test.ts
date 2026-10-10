import { describe, expect, it } from "vitest";

import { ConfigurationError, DownloadError, deserializeError, serializeError } from "../../src/errors.js";

describe("errors across a boundary", () => {
  it("carry code, message, details and whether a retry may help", () => {
    const wire = serializeError(new ConfigurationError("'emotion' needs a model", { details: { capability: "emotion" } }));
    expect(wire).toEqual({ type: "ConfigurationError", code: "ConfigurationError", message: "'emotion' needs a model", details: { capability: "emotion" }, retryable: false });
    expect(serializeError(new DownloadError("timed out")).retryable).toBe(true);
    const back = deserializeError(JSON.parse(JSON.stringify(wire)));
    expect(back).toBeInstanceOf(ConfigurationError);
    expect(back.details).toEqual({ capability: "emotion" });
  });

  it("still reads a payload from before the code field", () => {
    expect(deserializeError({ type: "DownloadError", message: "old peer" })).toBeInstanceOf(DownloadError);
  });
});
