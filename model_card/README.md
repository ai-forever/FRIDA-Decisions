---
license: mit
language:
- ru
- en
base_model: ai-forever/FRIDA
base_model_relation: finetune
pipeline_tag: text-classification
library_name: frida-decisions
tags:
- decisions
- routing
- intent-classification
- guardrails
- zero-shot-classification
- reranking
- encoder
- russian
---

# FRIDA-Decisions

[![Habr article](https://img.shields.io/badge/Habr-article%20(RU)-65A3BE?logo=habr&logoColor=white)](https://habr.com/ru/companies/sberbank/articles/1092056/) [![Code on GitHub](https://img.shields.io/badge/GitHub-FRIDA--Decisions-181717?logo=github&logoColor=white)](https://github.com/ai-forever/FRIDA-Decisions) [![Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Demo-Spaces-FFD21E)](https://huggingface.co/spaces/ai-forever/FRIDA-Decisions) [![Benchmark](https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark-razvilka-FFD21E)](https://huggingface.co/datasets/artemsnegirev/razvilka) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/notebooks/quickstart.ipynb) [![vLLM server](https://img.shields.io/badge/vLLM-server-30A2FF)](#vllm-server) [![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/ai-forever/FRIDA-Decisions/blob/main/LICENSE)

**FRIDA-Decisions** makes structured decisions over Russian text in a single encoder pass: pick one of K options, place a text on an ordinal scale, answer yes / no, or rank candidates. The options are written as text inside the request, so a new label set is a new JSON, not a new training run. No generation, no output tokens, no parsing: every answer is one of the declared options, with its confidence.

It is built on [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (T5 encoder, 823M parameters) and runs on a consumer GPU. Developed by УЭСМО, SberAI.

* **razvilka.** 0.893 on [razvilka](https://huggingface.co/datasets/artemsnegirev/razvilka) (735 items); TypeSafe Jev, a commercial API, scores 0.897 on the same items (paired McNemar p = 0.84). The highest among the open models we ran on razvilka.
* **Fast.** 28–34 ms per request on an RTX 5060 Ti (a ≈400-token text, 1–3 questions), in process; conditions in the latency table below.
* **Light.** 1.8 GiB allocated by PyTorch at peak over the whole razvilka run, plus the CUDA context; an int8 ONNX build runs on CPU, and an MLX build on Apple Silicon.
* **Packing.** All options of all questions share one sequence and the text is encoded once; with the state cache a follow-up question about the same text costs only its own tokens. A catalog of 243 intents is answered in 0.44 s, against 4.65 s for one sequence per option.
* **Serving.** A [vLLM server](#vllm-server) batches requests from many users and keeps the texts it has read in its prefix cache: about 60 requests/s on one RTX 5060 Ti with 8 requests in flight.

## Quickstart

```bash
pip install "frida-decisions[torch] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.4.0"
```

```python
from frida_decisions import Judge

judge = Judge.from_pretrained("ai-forever/FRIDA-Decisions", device="cuda")

request = {
    "state": "Здравствуйте, у меня не приходит код подтверждения уже час.",
    "questions": {
        "topic": {
            "type": "choice",
            "instructions": "К какой теме относится обращение?",
            "criteria": {
                "login": "вход в аккаунт, коды подтверждения, пароли",
                "payment": "оплата, списания, возвраты",
                "delivery": "доставка заказа",
            },
        },
        "urgency": {
            "type": "score",
            "instructions": "Оцени срочность обращения.",
            "criteria": ["низкая", "средняя", "высокая"],
        },
        "spam": {
            "type": "noul",
            "instructions": "Является ли сообщение спамом?",
            "criteria": {"true": "реклама, чужие ссылки, просьба перевести деньги",
                         "false": "вопрос или жалоба по нашему сервису"},
        },
    },
}
print(judge.judge(request)["answers"])
```

CPU without PyTorch: `pip install "frida-decisions[onnx] @ git+..."` and `OnnxJudge.from_pretrained("ai-forever/FRIDA-Decisions")` — int8 weights and per-token int8 activations; it scores 0.891 on razvilka (the same decision as the GPU model on 726 of 735 items), and a 384-token request with 3 questions takes about 0.9 s on 6 CPU threads, roughly 2.5x faster than fp32.

Notebooks: quickstart [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/notebooks/quickstart.ipynb) · evaluation on razvilka [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/benchmarks/razvilka/run_razvilka.ipynb)

## vLLM server

For many users at once (Linux, GPU). The server batches requests together and keeps the texts it has read in vLLM's prefix cache, so a follow-up question about a text costs only its own tokens.

```bash
pip install "frida-decisions[vllm] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.4.0"
vllm serve ai-forever/FRIDA-Decisions \
  --hf-overrides '{"architectures": ["FridaDecisionsModel"]}' \
  --io-processor-plugin frida_decisions \
  --no-enable-chunked-prefill --enforce-eager --max-model-len 2048
```

`POST /pooling` with the request under `data` returns the same response as `Judge` ([client example](https://github.com/ai-forever/FRIDA-Decisions/blob/v0.4.0/examples/vllm_client.py)).

| one RTX 5060 Ti, bf16, vLLM 0.29 | |
|---|--:|
| one request: a short ticket to a ≈400-token text, 1–3 questions | 25–45 ms |
| a follow-up question about a text the server has already read | 25–35 ms |
| one request choosing among 243 intents | ≈0.33 s |
| 8 requests in flight, razvilka-sized requests (≈260 tokens) | ≈60 requests/s |
| 8 requests in flight, short tickets (≈190 tokens) | ≈80 requests/s |
| razvilka, 735 items (`FRIDA_DECISIONS_STATE_MAX=512`, as in the PyTorch run) | 0.890 (PyTorch bf16: 0.893) |

Measured with the command above plus `--gpu-memory-utilization 0.45` (the card also drives a display), client on the same machine, after warm-up; the first request after a start takes about 0.3 s. vLLM and PyTorch differ on 2 razvilka items, both near-ties in fp32.

**Async API, no server.** `await judge.judge(request)` inside your own program: `AsyncJudge` wraps the PyTorch (or ONNX, or MLX) judge on any OS, `VllmJudge` runs the vLLM engine in your process (Linux, GPU) with the same batching and cache as the server ([example](https://github.com/ai-forever/FRIDA-Decisions/blob/v0.4.0/examples/async_judge.py)). Which is faster depends on the load: on the same card, a burst of 256 short tickets runs at ≈187 requests/s with `AsyncJudge` over PyTorch and ≈126 with `VllmJudge`; razvilka's mix of lengths with 32 requests in flight, at ≈58 and ≈69 (`VllmJudge` with 45 % of the GPU, medians of alternating runs).

**Apple Silicon.** `MlxJudge` (`frida-decisions[mlx]`, contributed by [@akolotov](https://github.com/akolotov)) runs the model in MLX, without torch. On an Apple M4 Pro, in float32 it makes the same 735 razvilka decisions as PyTorch float32 on CPU; in bfloat16, 733 of them, with the margin error of bfloat16 in PyTorch ([example](https://github.com/ai-forever/FRIDA-Decisions/blob/v0.4.0/examples/mlx_quickstart.py)). PyTorch itself also runs on a Mac with `device="mps"`.

## Question types

| type | asks | returns |
|---|---|---|
| `choice` | which of K options is right | the option key and a distribution over all options |
| `score` | where the text sits on an ordinal scale | a level 0..K−1 and its distribution |
| `noul` | is a statement true | `p(true)` |
| `ranking` | order candidates for a query | the order, a margin per candidate (`scores`) and a distribution |

### Filtering `ranking` candidates by a threshold

A `ranking` answer carries both `probabilities` — a softmax over the list, which splits one unit among the candidates and therefore cannot filter — and `scores`, the raw margin of each candidate, computed independently of the others. Use `scores` for top-k or a threshold.

Their level, though, is not calibrated: `ranking` is trained with a listwise softmax, which depends only on differences inside one list, so nothing pins where a particular request's margins sit. On the relevance task of razvilka (50 queries, 5 candidates, one relevant and four annotated negatives) the level ranges −1.88 to +10.84 with sd 2.32, against a median gap of +4.92 between the relevant candidate and the best irrelevant one. A single global threshold therefore reaches AUC 0.923 and F1 0.769, while the ordering inside a request is far better than that (top-1 0.900, AUC 0.950).

Subtracting the request's own candidates recovers most of the loss, because in a retriever's top-k most candidates are irrelevant and their margins are where that request's "irrelevant" sits. The held-out column fits the threshold on four fifths of the queries and scores it on the rest:

| threshold on | AUC (95 % CI) | best single threshold | held-out F1 |
|---|---|---|---|
| `scores` as they are | 0.923 (0.882–0.959) | +6.76 → F1 0.769 | 0.752 |
| `scores` minus a fixed anchor candidate | 0.929 (0.890–0.964) | +7.18 → F1 0.774 | 0.732 |
| `scores` minus the median of the request's own candidates | 0.951 (0.910–0.982) | +4.40 → F1 0.835 | 0.796 |
| `scores` minus the mean of the request's own candidates | 0.972 (0.943–0.992) | +3.52 → F1 0.882 | 0.855 |

```python
scores = judge(request)["answers"]["best"]["scores"]
level = sum(scores.values()) / len(scores)
relevant = [key for key, score in scores.items() if score - level >= 3.5]
```

An anchor candidate — one extra candidate whose text says it is not an answer — does not work: its own margin hardly moves with the request (sd 0.65, correlation with the request's level +0.22), so subtracting it removes noise rather than the offset.

Three things the recipe does not do. **+3.5 is not a constant**: it is fitted on lists of five with one relevant candidate, and kept unchanged on shorter lists it loses recall (three candidates F1 0.837, two 0.689), because the relevant candidate is part of the mean; refitted, those reach 0.917 and 0.953. **It cannot say that nothing is relevant** — with the relevant candidate dropped, 5.5 % of the remaining negatives still clear +3.52 (7.0 % clear the raw +6.76); for an absolute per-candidate probability ask `noul` per candidate, which is trained with a pairwise objective and returns a probability in 0…1. **Lists with several relevant candidates are not measured**: this task has exactly one, so the numbers describe a pool that is 20 % relevant, over 50 queries. Reproduce with [`tools/ranking_threshold.py`](https://github.com/ai-forever/FRIDA-Decisions/blob/main/tools/ranking_threshold.py).

### Input length

* **The 512-token range is for the text (`state`) only.** Questions and options do not count against it: each question's instructions take up to 96 tokens and each option up to 256. Options are packed into rows of up to 16 options / 1,024 tokens next to the text, and a request with more options simply gets more rows, so the number of options is limited only by time and memory.
* **Default cut: 384 tokens.** `Judge.from_pretrained(..., state_max=512)` uses the whole trained range (the razvilka numbers above use 512). A longer text is cut from the end, and the response says so in `usage.state_truncated`. On the vLLM server the cut is set with `FRIDA_DECISIONS_STATE_MAX`.
* **Longer texts.** T5's relative positions accept a longer `state_max`, but the model was trained on texts up to 512 tokens, and on longer ones accuracy may drop, especially for questions about details deep in the text. For long documents, split them into fragments of up to 512 tokens (by paragraph or section) and ask the questions per fragment: the fragments go in one `judge_batch` call, and a yes/no question like "is there X anywhere" becomes the maximum over fragments.
* **Cost grows with text + options, not text × options**: the text is encoded once per request, and with the state cache a follow-up question about the same text costs only its own tokens.

**Raising the limits.** The caps are settings, not part of the weights:

```python
judge = Judge.from_pretrained("ai-forever/FRIDA-Decisions", state_max=1024)  # text cut; the same for OnnxJudge
judge.config.option_max_tokens = 512        # one option (default 256)
judge.config.instruction_max_tokens = 192   # one question's instructions (default 96)
judge.config.max_row_tokens = 2048          # one packed row (default 1024)
judge.config.max_options_per_row = 32       # options per row (default 16)
```

The defaults ship with the weights in `decisions_config.json`; edit it in a local copy of the model folder to change them for every backend. For the vLLM server, point `vllm serve` at that folder, set `FRIDA_DECISIONS_STATE_MAX` for the text cut, and keep `--max-model-len` above the longest row (the server says so when a row does not fit). Larger values cost time and memory, and the model was trained within the defaults, so check quality on your own data before relying on them.

## Benchmarks

**razvilka** — 735 Russian items, 15 tasks (routing, intents, topic and sentiment classification, moderation, relevance ranking), all four question types, gold from published datasets. Every model answers the same items, each in its own input format, and all answers are scored by the same rule ([razvilka_eval.py](https://github.com/ai-forever/FRIDA-Decisions/tree/v0.4.0/benchmarks/razvilka)).

| model | parameters | accuracy |
|---|--:|--:|
| TypeSafe Jev (commercial API) | — | 0.897 |
| Qwen/Qwen3.5-35B-A3B, an LLM as a classifier² | 35B (3B active) | 0.897 |
| **FRIDA-Decisions** | 823M | **0.893** |
| FRIDA-Decisions, int8 ONNX on CPU | 823M | 0.891 |
| openai/gpt-oss-20b, an LLM as a classifier² | 21B (3.6B active) | 0.856 |
| smolnikov/migom-2b | 1.9B | 0.853 |
| Mapika/decider-2b | 1.9B | 0.833 |
| smolnikov/kivok-0.3b | 0.3B | 0.619 |
| fastino/GLiNER2.5-multi-Decide | 287M | 0.576 |
| convaiinnovations/laya (multilingual) | 322M | 0.559 |
| KaLM-Reranker-V1-Nano-R2 | 786M | 0.521 |
| open-jev (DeBERTa-v3-large) | 437M | 0.490¹ |
| lexical baseline | — | 0.333 |
| chance | — | 0.257 |

¹ 140 of 735 texts exceed open-jev's input window and get no answer from it and count as ties; on the other 595 it scores 0.565.

² Zero-shot through OpenRouter: one call per question at temperature 0, the text, question and options in the prompt, the chosen option key returned as JSON; reasoning off for Qwen3.5, `low` for gpt-oss, which cannot switch it off. Against FRIDA-Decisions (paired McNemar): Qwen3.5-35B-A3B p = 0.84, gpt-oss-20b p = 0.02.

**Latency**, one request, single stream, in process:

| hardware | request | time |
|---|---|--:|
| RTX 5060 Ti, bf16 | ≈400-token state, 1 question (3 options) | 28.2 ms |
| RTX 5060 Ti, bf16 | ≈400-token state, 3 questions (8 options) | 34.0 ms |
| Apple M4 Pro, MLX bf16 | ≈400-token state, 1 question (3 options) | 113 ms |
| Apple M4 Pro, MLX bf16 | ≈400-token state, 3 questions (8 options) | 139 ms |
| CPU, 6 threads, PyTorch fp32 | 384-token state, 3 questions (8 options) | 2.28 s |
| CPU, 6 threads, ONNX int8 | 384-token state, 3 questions (8 options) | 0.88 s |

GPU and Apple rows: median of 30 requests after warm-up, timing parsing, tokenisation, packing and the forward pass (Apple: `tools/mlx_check.py`, one run; MLX float32 takes 126 and 156 ms, PyTorch on MPS in bf16 139 and 168 ms). CPU rows were measured on a machine with background load; the ratio between them (about 2.5x) is the stable part.

**Packing**, RTX 5060 Ti, bf16, same model, one sequence per option vs packed with the state cache:

| scenario | candidates | tokens, naive / packed | time, naive / packed |
|---|--:|--:|--:|
| intent from a 243-intent support-bot catalog | 243 | 97,685 / 4,871 | 4.65 s / 0.44 s |
| full triage of a support email | 16 questions, 65 options | 23,487 / 1,853 | 1.10 s / 0.18 s |
| follow-up question about a cached document | 13 | 5,081 / 277 | 0.24 s / 25 ms |

## Training

* 1.42M examples (1.5M questions) over 151 question types; by our grouping they fall into eight domains: relevance and RAG grounding, NLI and fact checking, moderation and safety, agents and tool choice, topic classification, sentiment and emotion, LLM request routing, intents and customer support.
* 71% Russian, 29% English; mostly human or naturally labelled data, roughly a quarter (our estimate) with LLM-generated text or model labels; instruction wordings augmented with paraphrases.
* LoRA rank 16 on the attention projections (q, k, v, o) of all 24 layers plus a scalar head, 4.7M trainable parameters; merged into the weights in this release. Listwise softmax for `choice` / `ranking`, pairwise BCE for `noul`.
* One epoch, with a compute budget comparable to about 50 hours of a single consumer GPU (RTX 5060 Ti class).

## Files

* `model.safetensors` — encoder weights, bf16
* `head.safetensors` — readout head, fp32
* `decisions_config.json` — packing limits and instruction suffixes
* `onnx/model_int8_pertoken.onnx` — CPU build

## License

MIT. Based on [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (MIT).

## Citation

```bibtex
@misc{frida_decisions_2026,
  title  = {FRIDA-Decisions},
  author = {{УЭСМО, SberAI}},
  year   = {2026},
  url    = {https://github.com/ai-forever/FRIDA-Decisions}
}
```
