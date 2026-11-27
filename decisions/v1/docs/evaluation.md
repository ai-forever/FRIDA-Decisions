# Evaluation and tests

Run the commands below from `decisions/v1/`.

## Measured results

The checks live in [`tests/`](../tests) and [`tools/`](../tools) (results are written locally to `tests/_results/`, which is not committed); CPU and float32 unless noted. The 36-request set is the 9 requests in `tests/` plus 27 longer demo requests (routing, tool choice, moderation, support triage, a 243-intent catalog, 100-passage ranking); of those, only the intent catalog ships in `examples/data/`.

| check | result |
|---|---|
| exported model vs the training-time implementation (unmerged LoRA adapter) | 122/122 same decisions on 36 requests, max margin drift 3.8e-06 |
| released bf16 weights (run in float32) vs the same | 122/122 same decisions, max margin drift 0.045 |
| state cache vs packed rows | same decisions on 36 requests, max margin drift 6.7e-06 |
| packed rows vs one sequence per option | max margin drift 5.7e-06 on 9 requests |
| ONNX int8 vs PyTorch float32 | 120/122 same decisions (98.4%), max margin drift 0.65 |
| ONNX int8 (CPU) vs PyTorch bf16 (GPU) on razvilka | 726/735 same decisions; accuracy 0.891 vs 0.893 (`tools/release_eval.py`) |
| CPU latency, one request: 384-token state, 3 questions (8 options), 6 threads, background load | PyTorch fp32 2.28 s, ONNX int8 0.88 s (about 2.5x), `tools/cpu_latency_ab.py` |
| GPU latency, one request: ≈400-token state, RTX 5060 Ti, bf16 | 28.2 ms with 1 question, 34.0 ms with 3 questions (`tools/release_eval.py`) |
| peak GPU memory allocated by PyTorch over the razvilka run (without the CUDA context) | 1.8 GiB |
| vLLM backend arithmetic (CPU, float32, simulated paged cache) vs PyTorch float32 | same decisions on 9 requests, cold and with the state from the cache, max margin drift 5.2e-06 (`tests/test_vllm_backend.py`) |
| vLLM server (bf16, GPU) vs PyTorch bf16 (GPU) on razvilka | accuracy 0.890 vs 0.893 (654 vs 656 of 735, `tools/release_eval.py --vllm`): the 2 items where they differ are near-ties in float32 (top-two margin gaps 0.053 and 0.008), and PyTorch bf16 lands on the float32 side. The margin error from float32 is the same size for both over all 3,762 options — median 0.017 / 0.017, p95 0.066 / 0.067, p99 0.108 / 0.102 (vLLM / PyTorch) — and the same with the text read from the cache |
| vLLM server throughput, requests in flight | razvilka (735 requests, ≈260 tokens each), 8 in flight: 60.8 requests/s; short tickets (640 requests, ≈190 tokens, `examples/vllm_client.py --burst 640`): 81 with 8 in flight, 90 with 16 |
| vLLM server latency, one client over HTTP: ≈400-token state, RTX 5060 Ti, bf16 | 40 ms with 1 question, 44 ms with 3; 34 / 30 ms when the text is already cached |
| async API, RTX 5060 Ti, bf16, in a vLLM 0.29 container, default settings, `VllmJudge` with 45 % of the GPU, every run with a cold cache (`tools/async_bench.py`) | 256 short tickets (≈190 tokens) awaited at once: `AsyncJudge` over PyTorch 187 requests/s (186–189 over 5 runs), `VllmJudge` 126 (125–127); razvilka, 735 items, 32 in flight: 58 (58–58 over 3 runs) and 69 (69–69). Both make the same decisions as PyTorch float32 on the test requests (15/15, `tools/async_check.py`) |
| MLX vs PyTorch float32 (CPU) on razvilka, Apple M4 Pro, MLX 0.32.3 (`tools/mlx_check.py`) | float32: 735/735 same decisions, max margin drift 2.9e-05; bfloat16: 733/735, accuracy 0.891 vs 0.894, margin error median 0.018, p99 0.108 over 3,762 options (PyTorch bf16 on a GPU: 0.017 / 0.102); PyTorch on MPS, bfloat16: 735/735, margin error 0.017 / 0.108. MLX state cache vs packed rows on the intent catalog: max drift 1.2e-05 (float32). Under `AsyncJudge` (worker thread, 14 requests batched), margins within 1.2e-05 of direct calls; a batched call scores the catalog through packed rows, not the state cache |
| latency on Apple M4 Pro, one request: ≈400-token state (`state_max=512`), median of 30 after warm-up (`tools/mlx_check.py`) | MLX float32 126 ms with 1 question, 156 ms with 3; MLX bfloat16 113 / 139 ms; PyTorch on MPS 139 / 168 ms in bfloat16, 152 / 186 ms in float32 |
| peak MLX memory over the whole run above (load, razvilka, intent catalog, latency), `rows_per_forward=1` | 4.6 GiB in float32, 2.2 GiB in bfloat16 |
| a single threshold on `ranking` margins, relevance task of razvilka (50 queries, 5 candidates, `tools/ranking_threshold.py`) | pooled AUC 0.923 (bootstrap 95 % 0.882-0.959), best single threshold +6.76 -> F1 0.769, held out 0.752. Minus the mean of the request's own candidates: 0.972 (0.943-0.992), +3.52 -> 0.882, held out 0.855; minus the median 0.951 / 0.835 / 0.796; minus a fixed anchor candidate 0.929 / 0.774 / 0.732. The anchor's own margin has sd 0.65 and correlates +0.22 with the request's level (sd 2.32), which is why it removes nothing. The fitted threshold does not carry to shorter lists (five candidates 0.882, three 0.837, two 0.689) and does not recognise a list with nothing relevant (5.5 % of the negatives clear it) |
| accuracy on razvilka (735 items) | 0.893 (PyTorch bf16, GPU), see the model card |

GPU parity with the CPU path was smoke-tested on 15 decisions (`tools/gpu_parity.py`, all equal); the razvilka run above is the larger check.

## Tests

```bash
pip install -e ".[dev]"            # torch, transformers, onnx, onnxruntime, pytest
python tools/export_model.py --checkpoint <adapter folder> --out _export/FRIDA-Decisions
python tools/export_onnx.py --model-dir _export/FRIDA-Decisions
pytest -s
```

Model tests are skipped when no exported folder is present (`FD_MODEL_DIR`). The MLX tests (`tests/test_mlx*.py`) need `pip install -e ".[dev,mlx]"` and run on Apple Silicon; `tools/mlx_check.py` runs the razvilka comparison and the latency above. The vLLM backend's own tests run on CPU without vLLM; a running server is checked against `Judge` with

```bash
python tools/vllm_parity.py --reference-only          # float32 reference on CPU, before starting the server
python tools/vllm_parity.py --url http://127.0.0.1:8000
```
