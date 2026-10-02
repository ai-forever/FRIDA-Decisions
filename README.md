# FRIDA-Decisions

**FRIDA-Decisions answers structured questions about a text — pick an option, place it on a scale, say yes or no, rank candidates — and returns probabilities, not generated text.** All questions about one text, and all their options, are scored together by the [FRIDA](https://huggingface.co/ai-forever/FRIDA) encoder in packed sequences — the text is never re-encoded per option, and there is no decoding, no output tokens, no answer parsing.

* **Four question types** in one request: `choice`, `score`, `noul` (yes/no) and `ranking`.
* **Packed, not repeated.** Questions and options sit next to one copy of the text; a typical request is a single encoder call, and a 243-intent catalog is still one request.
* **Order-independent by construction.** Options never attend to each other, so adding, removing or reordering options does not change any other option's margin.
* **Exact state cache.** The text is encoded once and its keys/values are reused across rows and repeated requests — the same margins, less compute.
* **GPU or CPU.** PyTorch on GPU or CPU, an int8 ONNX model for CPU that does not need torch, or a vLLM server.

## Install

```bash
# PyTorch backend (GPU or CPU)
pip install "frida-decisions[torch] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.2.0"
# int8 ONNX backend for CPU, without torch
pip install "frida-decisions[onnx] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.2.0"
# vLLM server (GPU, Linux)
pip install "frida-decisions[vllm] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.2.0"
```

The core package needs only `numpy`, `tokenizers`, `safetensors` and `huggingface_hub`; each backend comes with its extra (`[torch]`, `[onnx]`, `[vllm]`). Python 3.10+. The weights are downloaded from the Hugging Face Hub on first use.

## Quickstart

```python
from frida_decisions import Judge

judge = Judge.from_pretrained("ai-forever/FRIDA-Decisions")   # GPU if available, else CPU

response = judge({
    "state": "Добрый день. Третий день не могу войти в личный кабинет: пишет, что пароль неверный, "
             "а письмо для сброса не приходит. Из-за этого не могу скачать счёт, срок оплаты завтра.",
    "questions": {
        "team":  {"type": "choice", "instructions": "В какую команду направить обращение?",
                  "criteria": {"auth": "доступ к аккаунту, вход, пароли",
                               "billing": "счета, оплата, возвраты",
                               "shipping": "доставка заказов"}},
        "angry": {"type": "noul", "instructions": "Автор раздражён?"},
    },
})
response["answers"]["team"]["choice"]     # "auth"
response["answers"]["angry"]["noul"]      # probability of "yes"
```

Several independent requests (different texts) can go in one call; you get one response per request, in the same order, with the same answers as calling `judge` on each:

```python
responses = judge.judge_batch([request_1, request_2, request_3])
```

More in [`examples/quickstart.py`](examples/quickstart.py) and the notebook [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) — run it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/notebooks/quickstart.ipynb)

## Request and response

A request is a `state` — a string or any JSON value (objects are rendered as compact, key-sorted JSON) — and named `questions`. Each question has a `type`, free-text `instructions` and, depending on the type, `criteria`. Every response carries `answers` (one per question), the raw `margins` of every option, and `usage`.

The answers below are real outputs of the released model (float32, CPU), trimmed to three decimals.

### `choice` — pick one of several described options

```json
{"state": "Здравствуйте! Вчера оплатил заказ №5512 картой, деньги списались дважды. Верните, пожалуйста, лишнее списание.",
 "questions": {"topic": {"type": "choice", "instructions": "Какая тема обращения?",
   "criteria": {"billing": "оплата, списания, возвраты денег",
                "delivery": "доставка и сроки получения заказа",
                "account": "вход в аккаунт, пароль, личные данные",
                "product": "качество и характеристики товара"}}}}
```
```json
{"answers": {"topic": {"type": "choice", "probabilities": {"billing": 0.999, "delivery": 0.001, "account": 0.0, "product": 0.001}, "confidence": 0.992, "choice": "billing"}}}
```

An option is shown to the model as `"<id>: <description>"`, so keep ids short and neutral and put the meaning in the description.

### `score` — place the input on an ordered scale

```json
{"state": "Сервер с базой заказов не отвечает уже сорок минут, клиенты не могут оформить покупку, каждая минута — потерянные продажи.",
 "questions": {"urgency": {"type": "score", "instructions": "Оцени срочность обращения по шкале ниже.",
   "criteria": ["Очень низкая: работа не страдает, ответить можно когда угодно.",
                "Низкая: неудобство небольшое, ответить можно на этой неделе.",
                "Средняя: работа затруднена, но обходной путь есть.",
                "Высокая: работа заблокирована, обходного пути нет.",
                "Критическая: потери растут с каждым часом."]}}}
```
```json
{"answers": {"urgency": {"type": "score", "probabilities": {"0": 0.0, "1": 0.0, "2": 0.0, "3": 0.086, "4": 0.914}, "confidence": 0.816, "score": 3.913, "legend": {"0": "Очень низкая: …", "...": "…", "4": "Критическая: …"}}}}
```

`score` is the expected level (0 … N−1) under `probabilities`; `legend` maps levels back to their descriptions.

### `noul` — yes / no

```json
{"state": "Курьер опоздал на два часа и даже не позвонил. Больше заказывать у вас не буду.",
 "questions": {"complaint": {"type": "noul", "instructions": "Это жалоба?"}}}
```
```json
{"answers": {"complaint": {"type": "noul", "noul": 0.674}}}
```

`noul` is the probability of "yes". `criteria` is optional: `{"true": "...", "false": "..."}` replaces the default wording of either side.

### `ranking` — order candidates best first

```json
{"state": "Как удалить накипь из чайника?",
 "questions": {"best": {"type": "ranking", "instructions": "Какой фрагмент лучше всего отвечает на вопрос?",
   "criteria": {"p1": "Налейте в чайник воду с двумя ложками лимонной кислоты, вскипятите и оставьте на час, затем промойте.",
                "p2": "Электрический чайник нельзя погружать в воду целиком: это опасно для нагревательного элемента.",
                "p3": "Чай лучше заваривать водой температурой 85–95 градусов, а не крутым кипятком.",
                "p4": "Уксус тоже растворяет накипь: смешайте его с водой один к одному, прокипятите и тщательно сполосните."}}}}
```
```json
{"answers": {"best": {"type": "ranking", "ranking": ["p4", "p1", "p2", "p3"], "scores": {"p1": 4.408, "p2": 0.167, "p3": -0.065, "p4": 5.851}, "probabilities": {"p1": 0.19, "p2": 0.003, "p3": 0.002, "p4": 0.805}, "confidence": 0.625}}}
```

A ranking candidate is content (a passage, an agent action, a document), not a labelled option: its id is never shown to the model. `criteria` may also be a plain list; ids are then `"0"`, `"1"`, … `scores` are the raw margins — use them for top-k or a threshold; `probabilities` split one unit among the candidates.

Invalid requests raise `frida_decisions.RequestError`; `error.payload()` gives `{"error": {"type", "message", "field"}}`.

## How it works

The model is FRIDA (a T5 encoder) with a linear head: a candidate's margin is the head applied to the mean of its own token states. Instead of encoding `[text][question][option]` once per option, a request is packed into one sequence with a block mask:

```
 tokens:   [ state ......... ][ question 1 ][ opt A ][ opt B ][ opt C ][ question 2 ][ yes ][ no ]
 positions: 0 ............ S   S ...... S+q   S+q..    S+q..    S+q..   S ...... S+r   S+r.. S+r..

 state      -> sees the state only
 question   -> sees the state and itself
 option     -> sees the state, its own question and itself   (never another option)
```

Positions restart after each question, so every option sits exactly where it would sit if it were scored alone; the mask makes the rest identical too. Large catalogs are split into rows of at most 16 options / 1,024 tokens — exactly, since options are independent.

Because the state only ever attends to itself, its per-layer keys and values do not depend on the questions. The torch backend encodes the state once and runs the question/option rows against the cached keys/values. This is exact (float32 drift ≈ 6.7e-06), and it kicks in automatically when a request needs more than one row or the same text was seen recently.

For the 243-intent example in [`examples/data/intent_catalog.json`](examples/data/intent_catalog.json) the encoder reads 97,685 tokens if every intent gets its own sequence, 9,836 (16 rows) when packed, and 4,871 with the state cache.

## Backends

| backend | install | where | notes |
|---|---|---|---|
| `Judge` (PyTorch) | `[torch]` | GPU (bf16) or CPU (fp32) | packing + exact state cache |
| `OnnxJudge` (ONNX Runtime) | `[onnx]` | CPU | int8, packing via graph inputs, no state cache |
| vLLM server | `[vllm]` | GPU (bf16), Linux | packing + state cache through vLLM's prefix cache, continuous batching across users |

The ONNX model is for CPU only. Its weights are int8 (one scale per output channel) and its activations are quantised to int8 on the fly, with one scale per token, while the decision head stays in float32. Quantisation changes the margins: on the 122 decisions of the test set, the ONNX model agrees with PyTorch float32 on 120, and the largest margin difference is 0.65. Where every decision matters, use the PyTorch backend.

```python
from frida_decisions import OnnxJudge

judge = OnnxJudge.from_pretrained("ai-forever/FRIDA-Decisions", threads=8)
judge(request)          # same request and response format
```

### vLLM server

`[vllm]` installs a vLLM plugin: the model, its attention (the packed block mask and FRIDA's relative positions, which vLLM's own attention kernels do not support) and the request front end. vLLM finds it on start; there is nothing to import.

```bash
vllm serve ai-forever/FRIDA-Decisions \
  --hf-overrides '{"architectures": ["FridaDecisionsModel"]}' \
  --io-processor-plugin frida_decisions \
  --no-enable-chunked-prefill --enforce-eager --max-model-len 2048
```

In Docker (Linux, or Windows with WSL2):

```bash
docker run --gpus all --ipc=host -p 8000:8000 --entrypoint bash vllm/vllm-openai:v0.29.0 -c \
  'pip install "frida-decisions[vllm] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.2.0" && exec vllm serve ai-forever/FRIDA-Decisions --hf-overrides "{\"architectures\": [\"FridaDecisionsModel\"]}" --io-processor-plugin frida_decisions --no-enable-chunked-prefill --enforce-eager --max-model-len 2048'
```

A request is the same JSON as for `Judge`, under `data`; the response `data` is what `Judge` returns:

```python
import httpx

r = httpx.post("http://localhost:8000/pooling", json={"model": "ai-forever/FRIDA-Decisions", "data": request})
r.json()["data"]["answers"]
```

A fuller client — a follow-up question answered from the cache, the 243-intent catalog, a burst of concurrent requests — is [`examples/vllm_client.py`](examples/vllm_client.py) (standard library only).

* **GPU memory.** vLLM takes 90 % of the GPU by default; on a card that also drives a display pass `--gpu-memory-utilization` (0.45 on an 8 GB card leaves room for the desktop and holds about 9,700 tokens of cache).
* **Required flags.** The attention is bidirectional, so a text split across scheduler steps would be encoded without seeing its own end: the server refuses to start with chunked prefill on. `--kv-cache-dtype` must stay `auto`, because a cached text is reused bit for bit. `--enforce-eager` is expected (the attention is plain PyTorch).
* **State cache.** Prefix caching is on by default. A text the server has already read is not encoded again, whether the next request comes from the same user or another one; `usage.cached_tokens` shows it. Each row starts with a hash of the whole text, so a cached block is reused only for the same text, never for another text that starts the same way. `--no-enable-prefix-caching` turns the cache off; pass `cache_salt` in the request body to share cached texts only within one tenant.
* **Limits.** The packing limits come from `decisions_config.json`; the state is cut at 384 tokens, as in `Judge` (`FRIDA_DECISIONS_STATE_MAX` in the server's environment changes it).
* **Raw token ids.** Clients that tokenize themselves can send rows to `POST /pooling` with `"task": "classify"` and `"input": [[ids], ...]` and get each row's margins; `frida_decisions.vllm_backend.rows.build_rows` builds the rows. A row that breaks the format, or whose text hash is wrong, is refused with HTTP 400 and does not affect other requests.
* **Version.** Tested with vLLM 0.29.0. The plugin uses vLLM internals (attention backends, poolers) that change between releases; with another version the server logs a warning, and `tools/vllm_parity.py` checks a running server against `Judge`.
* **One difference from `Judge`.** Special-token strings inside the text (a literal `<s>` or `</s>`) are tokenized as plain text, because the plugin uses those tokens to mark the row layout; `usage.special_tokens_split` lists where that happened.

### Measured

The checks live in [`tests/`](tests) and [`tools/`](tools) (results are written locally to `tests/_results/`, which is not committed); CPU and float32 unless noted. The 36-request set is the 9 requests in `tests/` plus 27 longer demo requests (routing, tool choice, moderation, support triage, a 243-intent catalog, 100-passage ranking); of those, only the intent catalog ships in `examples/data/`.

| check | result |
|---|---|
| exported model vs the training-time implementation (unmerged LoRA adapter) | 122/122 same decisions on 36 requests, max margin drift 3.8e-06 |
| released bf16 weights (run in float32) vs the same | 122/122 same decisions, max margin drift 0.045 |
| state cache vs packed rows | same decisions on 36 requests, max margin drift 6.7e-06 |
| packed rows vs one sequence per option | max margin drift 5.7e-06 on 9 requests |
| ONNX int8 vs PyTorch float32 | 120/122 same decisions (98.4%), max margin drift 0.65 |
| ONNX int8 (CPU) vs PyTorch bf16 (GPU) on razvilka | 726/735 same decisions; accuracy 0.891 vs 0.893 (`tools/release_eval.py`) |
| CPU latency, one request: 384-token state, 3 questions (8 options), 6 threads, background load | PyTorch fp32 2.28 s, ONNX int8 0.88 s (about 2.5x), `tools/cpu_latency_ab.py` |
| GPU latency, one request: ~400-token state, RTX 5060 Ti, bf16 | 28.2 ms with 1 question, 34.0 ms with 3 questions (`tools/release_eval.py`) |
| peak GPU memory allocated by PyTorch over the razvilka run (without the CUDA context) | 1.8 GiB |
| vLLM backend arithmetic (CPU, float32, simulated paged cache) vs PyTorch float32 | same decisions on 9 requests, cold and with the state from the cache, max margin drift 5.2e-06 (`tests/test_vllm_backend.py`) |
| vLLM server (bf16, GPU) vs PyTorch bf16 (GPU) on razvilka | accuracy 0.890 vs 0.893 (654 vs 656 of 735, `tools/release_eval.py --vllm`): the 2 items where they differ are near-ties in float32 (top-two margin gaps 0.053 and 0.008), and PyTorch bf16 lands on the float32 side. The margin error from float32 is the same size for both over all 3,762 options — median 0.017 / 0.017, p95 0.066 / 0.067, p99 0.108 / 0.102 (vLLM / PyTorch) — and the same with the text read from the cache |
| vLLM server throughput, requests in flight | razvilka (735 requests, ~260 tokens each), 8 in flight: 60.8 requests/s; short tickets (640 requests, ~190 tokens, `examples/vllm_client.py --burst 640`): 81 with 8 in flight, 90 with 16 |
| vLLM server latency, one client over HTTP: ~400-token state, RTX 5060 Ti, bf16 | 40 ms with 1 question, 44 ms with 3; 34 / 30 ms when the text is already cached |
| accuracy on razvilka (735 items) | 0.893 (PyTorch bf16, GPU), see the model card |

GPU parity with the CPU path was smoke-tested on 15 decisions (`tools/gpu_parity.py`, all equal); the razvilka run above is the larger check.

## Limits

* **State:** up to 512 tokens is the recommended range — the range the model was trained on. `Judge.from_pretrained(..., state_max=384)` sets where the state is cut (default 384); longer states are cut from the end, and `usage.state_truncated` tells you when that happened.
* **Instructions:** up to 96 tokens per question, including a short type-specific suffix the model was trained with.
* **Options:** up to 256 tokens each. The number of options is limited only by time and memory; rows hold up to 16 options / 1,024 tokens and are split automatically.
* **Language:** FRIDA is a Russian–English encoder, and the built-in instruction suffixes are in Russian.

The packing limits ship with the weights in `decisions_config.json` and are read by every backend.

## Model folder

```
config.json               T5 encoder config
model.safetensors         encoder weights (bfloat16), LoRA already merged
head.safetensors          decision head, float32: weight (1, 1536), bias (1,)
tokenizer.json, tokenizer_config.json
decisions_config.json     packing limits, instruction suffixes, relative-position settings
onnx/model_int8_pertoken.onnx   int8 graph for OnnxJudge (1.17 GiB)
```

`tools/export_model.py` builds this folder from a FRIDA LoRA adapter (merge `W += B·A·α/r`, no `peft` needed at inference), and `tools/export_onnx.py` adds the int8 ONNX graph.

## Tests

```bash
pip install -e ".[dev]"            # torch, transformers, onnx, onnxruntime, pytest
python tools/export_model.py --checkpoint <adapter folder> --out _export/FRIDA-Decisions
python tools/export_onnx.py --model-dir _export/FRIDA-Decisions
pytest -s
```

Model tests are skipped when no exported folder is present (`FD_MODEL_DIR`). The vLLM backend's own tests run on CPU without vLLM; a running server is checked against `Judge` with

```bash
python tools/vllm_parity.py --reference-only          # float32 reference on CPU, before starting the server
python tools/vllm_parity.py --url http://127.0.0.1:8000
```

## License

MIT — see [LICENSE](LICENSE).
The base model [FRIDA](https://huggingface.co/ai-forever/FRIDA) is © ai-forever and released under the MIT License.

## Citation

```bibtex
@misc{frida_decisions_2026,
  title  = {FRIDA-Decisions},
  author = {TODO},
  year   = {2026},
  url    = {https://github.com/ai-forever/FRIDA-Decisions}
}
```
