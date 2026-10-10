# onnxweb-core

Run ONNX models in the browser: WebGPU, WebNN and WebAssembly, chosen per device.

The base the other `onnxweb-*` libraries build on:

- **Backends**: detects WebGPU, WebNN (NPU first) and WebAssembly, and falls back down the chain when a backend refuses a graph — at session creation or at its first run.
- **Model loading**: Hugging Face Hub or own server, ranged and resumable downloads into Cache Storage or OPFS, weight files over 2 GB taken in parts, room made by evicting old models.
- **Graph rewrites before the session**: merged decoders split, 4-bit matrices repacked for WebGPU, quantized embedding tables gathered before dequantization, inline weights moved out (the 4 GB WebAssembly heap).
- **Generation**: greedy, beam search and sampling, logits processors, the KV cache kept on the GPU.
- **Text**: tokenizers (`tokenizer.json`, SentencePiece), language identification, document splitting, exports (SRT, VTT, JSON, CSV…).
- **Workers**: a task protocol to run any of it off the main thread.

```ts
import * as ort from "onnxruntime-web";
import { huggingFaceResolver, loadModel } from "@youni/onnxweb-core";

const bundle = await loadModel({
  ort,
  resolver: huggingFaceResolver({ repo: "Xenova/bge-small-en-v1.5", subfolder: "onnx", manifest: false }),
  model: { name: "bge-small-en-v1.5", computeType: "float32" },
  backends: "webgpu,wasm",
});
const outputs = await bundle.only.run(feeds);
```

## Install

Build and test:

```sh
npm install
npm run build   # dist/
npm test        # unit tests (Node, onnxruntime-web on WebAssembly)
```

## Licence

Apache-2.0 for the code. Model weights keep their own licence (listed above where it matters).
