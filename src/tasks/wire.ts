/** The names a result carries when it crosses to Python. */

export function snakeCase(name: string): string {
  return name.replace(/([a-z0-9])([A-Z])/g, "$1_$2").replace(/([A-Z])([A-Z][a-z])/g, "$1_$2").toLowerCase();
}

export function camelCase(name: string): string {
  return name.replace(/_([a-z0-9])/g, (_, char: string) => char.toUpperCase());
}

function convert(
  value: unknown,
  rename: (key: string) => string,
  opaque: ReadonlySet<string>,
  path: string,
): unknown {
  if (value === null || typeof value !== "object" || ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    return value;
  }
  if (Array.isArray(value)) return value.map((item) => convert(item, rename, opaque, `${path}[]`));
  const out: Record<string, unknown> = {};
  const keepKeys = opaque.has(path);
  for (const [key, item] of Object.entries(value)) {
    const name = keepKeys ? key : rename(key);
    out[name] = convert(item, rename, opaque, path ? `${path}.${name}` : name);
  }
  return out;
}

/**
 * A JS value with Python field names. `opaque` lists the paths (in the OUTPUT's naming,
 * `first_step_probs` or `segments[].scores`) whose keys are data and must not be renamed.
 */
export function toPythonNames(value: unknown, opaque: readonly string[] = []): unknown {
  return convert(value, snakeCase, new Set(opaque), "");
}

/** The inverse: a Python payload with JS field names. */
export function fromPythonNames(value: unknown, opaque: readonly string[] = []): unknown {
  return convert(value, camelCase, new Set(opaque), "");
}
