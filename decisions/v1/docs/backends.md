# Inference backends

| backend | install | where | notes |
|---|---|---|---|
| `Judge` (PyTorch) | `[torch]` | GPU (bf16) or CPU (fp32) | packing + exact state cache |
| `OnnxJudge` (ONNX Runtime) | `[onnx]` | CPU | int8, packing via graph inputs, no state cache |
| `MlxJudge` (MLX) | `[mlx]` | Apple Silicon (fp32 or bf16) | packing + exact state cache, no torch |
| vLLM server | `[vllm]` | GPU (bf16), Linux | packing + state cache through vLLM's prefix cache, continuous batching across users |
| `AsyncJudge` | `[torch]`, `[onnx]` or `[mlx]` | wherever `Judge` / `OnnxJudge` / `MlxJudge` runs | `await judge.judge(request)`; requests awaited together share `judge_batch` calls |
| `VllmJudge` | `[vllm]` | GPU (bf16), Linux | `await judge.judge(request)` on the vLLM engine in your process: the server's batching and state cache, no HTTP |

The ONNX model is for CPU only. Its weights are int8 (one scale per output channel) and its activations are quantised to int8 on the fly, with one scale per token, while the decision head stays in float32. Quantisation changes the margins: on the 122 decisions of the test set, the ONNX model agrees with PyTorch float32 on 120, and the largest margin difference is 0.65. Where every decision matters, use the PyTorch backend.

```python
from frida_decisions import OnnxJudge

judge = OnnxJudge.from_pretrained("ai-forever/FRIDA-Decisions", threads=8)
judge(request)          # same request and response format
```

## MLX on Apple Silicon

`MlxJudge` runs the encoder in [MLX](https://github.com/ml-explore/mlx) on the Apple GPU, without torch, with the same packing and exact state cache as `Judge`. The encoder runs in float32 by default; `dtype=mlx.core.bfloat16` halves its memory, and the decision head stays in float32 either way. Contributed by [@akolotov](https://github.com/akolotov).

```python
from frida_decisions import MlxJudge

judge = MlxJudge.from_pretrained("ai-forever/FRIDA-Decisions")   # float32
judge(request)          # same request and response format
```

On razvilka (Apple M4 Pro, `tools/mlx_check.py`) MLX in float32 makes the same 735 decisions as PyTorch float32 on CPU; in bfloat16 it makes 733 of them, and its margin error against float32 (median 0.018, p99 0.108) is that of bfloat16 in PyTorch. A request with a ≈400-token state and 3 questions takes 156 ms in float32 and 139 ms in bfloat16.

* `rows_per_forward` (default 1) caps the packed rows per encoder call, and with it the attention memory; `None` runs all rows at once.
* `state_cache_mb` (default 512) bounds the state cache, 0 turns it off; `compile_encoder=False` turns off `mx.compile`.
* `AsyncJudge.from_pretrained(..., backend="mlx")` wraps it like the other judges.
* PyTorch also runs on a Mac: `Judge.from_pretrained(..., device="mps")` (bfloat16) makes the same 735 decisions on razvilka. MLX in bfloat16 takes 17–19 % less time on the requests above (113 against 139 ms with 1 question, 139 against 168 ms with 3; one run) and does not need torch.

More in [`examples/mlx_quickstart.py`](../examples/mlx_quickstart.py).

## vLLM server

`[vllm]` installs a vLLM plugin: the model, its attention (the packed block mask and FRIDA's relative positions, which vLLM's own attention kernels do not support) and the request front end. vLLM finds it on start; there is nothing to import.

```bash
vllm serve ai-forever/FRIDA-Decisions \
  --hf-overrides '{"architectures": ["FridaDecisionsModel"]}' \
  --io-processor-plugin frida_decisions \
  --no-enable-chunked-prefill --enforce-eager --max-model-len 2048
```

In Docker (Linux, or Windows with WSL2); the vLLM image has no `git`, so the package comes as the release archive:

```bash
docker run --gpus all --ipc=host -p 8000:8000 --entrypoint bash vllm/vllm-openai:v0.29.0 -c \
  'pip install "frida-decisions[vllm] @ https://github.com/ai-forever/FRIDA-Decisions/archive/refs/tags/v0.4.0.tar.gz" && exec vllm serve ai-forever/FRIDA-Decisions --hf-overrides "{\"architectures\": [\"FridaDecisionsModel\"]}" --io-processor-plugin frida_decisions --no-enable-chunked-prefill --enforce-eager --max-model-len 2048'
```

A request is the same JSON as for `Judge`, under `data`; the response `data` is what `Judge` returns:

```python
import httpx

r = httpx.post("http://localhost:8000/pooling", json={"model": "ai-forever/FRIDA-Decisions", "data": request})
r.json()["data"]["answers"]
```

A fuller client — a follow-up question answered from the cache, the 243-intent catalog, a burst of concurrent requests — is [`examples/vllm_client.py`](../examples/vllm_client.py) (standard library only).

* **GPU memory.** vLLM takes 90 % of the GPU by default; on a card that also drives a display pass `--gpu-memory-utilization` (0.45 on an 8 GB card leaves room for the desktop and holds about 9,700 tokens of cache).
* **Required flags.** The attention is bidirectional, so a text split across scheduler steps would be encoded without seeing its own end: the server refuses to start with chunked prefill on. `--kv-cache-dtype` must stay `auto`, because a cached text is reused bit for bit. `--enforce-eager` is expected (the attention is plain PyTorch).
* **State cache.** Prefix caching is on by default. A text the server has already read is not encoded again, whether the next request comes from the same user or another one; `usage.cached_tokens` shows it. Each row starts with a hash of the whole text, so a cached block is reused only for the same text, never for another text that starts the same way. `--no-enable-prefix-caching` turns the cache off; pass `cache_salt` in the request body to share cached texts only within one tenant.
* **Limits.** The packing limits come from `decisions_config.json`; the state is cut at 384 tokens, as in `Judge` (`FRIDA_DECISIONS_STATE_MAX` in the server's environment changes it).
* **Raw token ids.** Clients that tokenize themselves can send rows to `POST /pooling` with `"task": "classify"` and `"input": [[ids], ...]` and get each row's margins; `frida_decisions.vllm_backend.rows.build_rows` builds the rows. A row that breaks the format, or whose text hash is wrong, is refused with HTTP 400 and does not affect other requests.
* **Version.** Tested with vLLM 0.29.0. The plugin uses vLLM internals (attention backends, poolers) that change between releases; with another version the server logs a warning, and `tools/vllm_parity.py` checks a running server against `Judge`.
* **One difference from `Judge`.** Special-token strings inside the text (a literal `<s>` or `</s>`) are tokenized as plain text, because the plugin uses those tokens to mark the row layout; `usage.special_tokens_split` lists where that happened.

## Async API

`AsyncJudge` and `VllmJudge` give `async def judge(request) -> response` with the request and response of `Judge`:

```python
import asyncio
from frida_decisions import AsyncJudge, VllmJudge

async def main(requests):
    # PyTorch (or backend="onnx"): any OS, GPU or CPU
    async with AsyncJudge.from_pretrained("ai-forever/FRIDA-Decisions") as judge:
        response = await judge.judge(requests[0])

    # the vLLM engine inside this process (Linux, CUDA GPU), no server
    async with VllmJudge.from_pretrained("ai-forever/FRIDA-Decisions", gpu_memory_utilization=0.45) as judge:
        responses = await asyncio.gather(*(judge.judge(r) for r in requests))
```

More in [`examples/async_judge.py`](../examples/async_judge.py).

* **`AsyncJudge`** runs the model in one worker thread, so the event loop stays free while it computes. Requests awaited at the same time are scored together: while one batch runs, the next queues up and goes to `judge_batch` in one call (up to `max_batch`, 8 by default). It wraps any judge: `AsyncJudge(Judge.from_pretrained(...))`.
* **`VllmJudge`** starts the vLLM engine (in a process vLLM manages) with the flags the model needs and submits each request's rows to it. Requests from all coroutines are batched by vLLM's scheduler, and a text the engine has already read comes from its prefix cache; pass `cache_salt=` to `judge()` to share cached texts only within one tenant. Cancelling an awaiting `judge()` aborts its rows in the engine. Stop it with `close()` or `async with`.
* **Which one** depends on the load (RTX 5060 Ti, in a vLLM 0.29 container, both judges with default settings, `VllmJudge` with 45 % of the GPU; medians of alternating runs, `tools/async_bench.py`): on a burst of 256 short tickets of one length (≈190 tokens) `AsyncJudge` over PyTorch answers ≈187 requests/s against ≈126 for `VllmJudge`; on razvilka's mix of lengths with 32 requests in flight, ≈58 against ≈69. Likely, but not profiled: packed batches suit equal lengths and pad mixed ones, while vLLM schedules rows of any length. `VllmJudge` also keeps the texts it has read in vLLM's prefix cache. Both raise `RequestError` for an invalid request, as `Judge` does.
