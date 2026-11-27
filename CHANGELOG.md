# Changelog

## Unreleased

- The private V2 release adds text and image retrieval with [FRIDA-Embeddings V2](embeddings/v2/inference/) and structured decisions over text and images with [FRIDA-Decisions V2](decisions/v2/inference/). Decisions 0.5.0 includes the V1 SDK and both generations of the vLLM plugin.
- The repository now groups code and documentation by project. The [FRIDA-Decisions V1 package](decisions/v1/) lives in `decisions/v1/`. The Python import is still `frida_decisions`.
- The [FRIDA-Embeddings guide](embeddings/) covers the original FRIDA encoder and the V2 inference integrations.

## 2026-10-06 · FRIDA-Decisions 0.4.0

Added the MLX backend for Apple Silicon. [Release tag](https://github.com/ai-forever/FRIDA-Decisions/tree/v0.4.0)

## 2026-10-04 · FRIDA-Decisions 0.3.0

Added asynchronous inference with `AsyncJudge` and `VllmJudge`. [Release tag](https://github.com/ai-forever/FRIDA-Decisions/tree/v0.3.0)

## 2026-10-02 · FRIDA-Decisions 0.2.0

Added the vLLM backend for serving concurrent requests on GPU. [Release tag](https://github.com/ai-forever/FRIDA-Decisions/tree/v0.2.0)

## 2026-10-02 · FRIDA-Decisions 0.1.x

The initial release included PyTorch inference, an int8 ONNX CPU backend and the razvilka evaluation. Version 0.1.1 added float16 support for GPUs without native bfloat16. [0.1.0](https://github.com/ai-forever/FRIDA-Decisions/tree/v0.1.0) · [0.1.1](https://github.com/ai-forever/FRIDA-Decisions/tree/v0.1.1)
