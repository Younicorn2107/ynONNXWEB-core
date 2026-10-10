import { defineConfig } from "tsup";

/**
 * Three profiles from one source, because the same code is consumed three ways.
 *
 * NOTE: no profile sets `clean`. tsup runs multi-config entries in PARALLEL,
 * so a clean in one of them races the others and deletes files they have
 * already written — it silently cost `auto.d.ts` until a test asserted the
 * declared exports actually exist. `npm run build` clears dist/ first instead.
 *
 *  - `index` / `core` — for npm. onnxruntime-web stays EXTERNAL: the host app
 *    owns exactly one ORT instance, and bundling a second copy would give the
 *    page two wasm heaps whose `ort.env` writes land on the wrong one.
 *  - `bundle` — for the Python wheel and for a plain <script type="module">,
 *    neither of which has a bundler. Its entry is `auto.ts`, which is the one
 *    file in this package that IMPORTS onnxruntime-web rather than taking it
 *    as an argument; everything else stays injection-based so a host app can
 *    own the single runtime instance.
 *  - `worker` — inlined AND `noExternal`, deliberately. A worker script served
 *    from a CDN is cross-origin and `new Worker(url)` throws, so the client
 *    fetches the text and builds the worker from a Blob URL. Inside a blob
 *    worker `import.meta.url` is the blob URL, so any relative or dynamic
 *    import breaks. One self-contained file is the only shape that survives.
 */
export default defineConfig([
  {
    entry: { index: "src/index.ts", core: "src/core.ts" },
    format: ["esm"],
    dts: true,
    treeshake: true,
    external: ["onnxruntime-web"],
  },
  {
    entry: { "ynonnxweb.bundle": "src/auto.ts" },
    format: ["esm"],
    // `platform: "browser"` matters: the default injects a `createRequire`
    // shim from node:module, which a browser cannot resolve — the bundle
    // would fail on its first line in the one environment it exists for.
    platform: "browser",
    dts: { entry: { auto: "src/auto.ts" } },
    treeshake: true,
    // Everything inlined: the page that loads this has no bundler to resolve a bare import.
    noExternal: [/.*/],
  },
  {
    entry: { "ynonnxweb.worker": "src/worker/host.ts" },
    format: ["esm"],
    platform: "browser",
    dts: false,
    splitting: false,
    treeshake: true,
    noExternal: [/.*/],
  },
]);
