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

**[Code on GitHub](https://github.com/ai-forever/FRIDA-Decisions)** · **[Demo](https://huggingface.co/spaces/ai-forever/FRIDA-Decisions)** · **[Benchmark: razvilka](https://huggingface.co/datasets/artemsnegirev/razvilka)** · **[Colab quickstart](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/notebooks/quickstart.ipynb)** · **[vLLM server](#vllm-server)**

**FRIDA-Decisions** makes structured decisions over Russian text in a single encoder pass: pick one of K options, place a text on an ordinal scale, answer yes / no, or rank candidates. The options are written as text inside the request, so a new label set is a new JSON, not a new training run. No generation, no output tokens, no parsing: every answer is one of the declared options, with its confidence.

It is built on [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (T5 encoder, 823M parameters) and runs on a consumer GPU. Developed by УЭСМО, SberAI.

* **razvilka.** 0.893 on [razvilka](https://huggingface.co/datasets/artemsnegirev/razvilka) (735 items); TypeSafe Jev, a commercial API, scores 0.897 on the same items (paired McNemar p = 0.84). The highest among the open models we ran on razvilka.
* **Fast.** 28–34 ms per request on an RTX 5060 Ti (a ≈400-token text, 1–3 questions), in process; conditions in the latency table below.
* **Light.** 1.8 GiB allocated by PyTorch at peak over the whole razvilka run, plus the CUDA context; an int8 ONNX build runs on CPU.
* **Packing.** All options of all questions share one sequence and the text is encoded once; with the state cache a follow-up question about the same text costs only its own tokens. A catalog of 243 intents is answered in 0.44 s, against 4.65 s for one sequence per option.
* **Serving.** A [vLLM server](#vllm-server) batches requests from many users and keeps the texts it has read in its prefix cache: about 60 requests/s on one RTX 5060 Ti with 8 requests in flight.

## Quickstart

```bash
pip install "frida-decisions[torch] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.2.0"
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
pip install "frida-decisions[vllm] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.2.0"
vllm serve ai-forever/FRIDA-Decisions   --hf-overrides '{"architectures": ["FridaDecisionsModel"]}'   --io-processor-plugin frida_decisions   --no-enable-chunked-prefill --enforce-eager --max-model-len 2048
```

`POST /pooling` with the request under `data` returns the same response as `Judge` ([client example](https://github.com/ai-forever/FRIDA-Decisions/blob/v0.2.0/examples/vllm_client.py)).

| one RTX 5060 Ti, bf16, vLLM 0.29 | |
|---|--:|
| one request: a short ticket to a ≈400-token text, 1–3 questions | 25–45 ms |
| a follow-up question about a text the server has already read | 25–35 ms |
| one request choosing among 243 intents | ≈0.33 s |
| 8 requests in flight, razvilka-sized requests (≈260 tokens) | ≈60 requests/s |
| 8 requests in flight, short tickets (≈190 tokens) | ≈80 requests/s |
| razvilka, 735 items (`FRIDA_DECISIONS_STATE_MAX=512`, as in the PyTorch run) | 0.890 (PyTorch bf16: 0.893) |

Measured with the command above plus `--gpu-memory-utilization 0.45` (the card also drives a display), client on the same machine, after warm-up; the first request after a start takes about 0.3 s. vLLM and PyTorch differ on 2 razvilka items, both near-ties in fp32.

## Question types

| type | asks | returns |
|---|---|---|
| `choice` | which of K options is right | the option key and a distribution over all options |
| `score` | where the text sits on an ordinal scale | a level 0..K−1 and its distribution |
| `noul` | is a statement true | `p(true)` |
| `ranking` | order candidates for a query | the order and a margin per candidate |

### Input length

* **The 512-token range is for the text (`state`) only.** Questions and options do not count against it: each question's instructions take up to 96 tokens and each option up to 256. Options are packed into rows of up to 16 options / 1,024 tokens next to the text, and a request with more options simply gets more rows, so the number of options is limited only by time and memory.
* **Default cut: 384 tokens.** `Judge.from_pretrained(..., state_max=512)` uses the whole trained range (the razvilka numbers above use 512). A longer text is cut from the end, and the response says so in `usage.state_truncated`. On the vLLM server the cut is set with `FRIDA_DECISIONS_STATE_MAX`.
* **Longer texts.** T5's relative positions accept a longer `state_max`, but the model was trained on texts up to 512 tokens, and on longer ones accuracy drops sharply, especially for questions about details deep in the text. For long documents, split them into fragments of up to 512 tokens (by paragraph or section) and ask the questions per fragment: the fragments go in one `judge_batch` call, and a yes/no question like "is there X anywhere" becomes the maximum over fragments.
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

**razvilka** — 735 Russian items, 15 tasks (routing, intents, topic and sentiment classification, moderation, relevance ranking), all four question types, gold from published datasets. Every model answers the same items, each in its own input format, and all answers are scored by the same rule ([razvilka_eval.py](https://github.com/ai-forever/FRIDA-Decisions/tree/v0.2.0/benchmarks/razvilka)).

| model | parameters | accuracy |
|---|--:|--:|
| TypeSafe Jev (commercial API) | — | 0.897 |
| **FRIDA-Decisions** | 823M | **0.893** |
| FRIDA-Decisions, int8 ONNX on CPU | 823M | 0.891 |
| smolnikov/migom-2b | 1.9B | 0.853 |
| Mapika/decider-2b | 1.9B | 0.833 |
| smolnikov/kivok-0.3b | 0.3B | 0.619 |
| fastino/GLiNER2.5-multi-Decide | 287M | 0.576 |
| convaiinnovations/laya (multilingual) | — | 0.559 |
| KaLM-Reranker-V1-Nano-R2 | 786M | 0.521 |
| open-jev (DeBERTa-v3-large) | 437M | 0.490¹ |
| lexical baseline | — | 0.333 |
| chance | — | 0.257 |

¹ 140 of 735 texts exceed open-jev's input window and get no answer from it and count as ties; on the other 595 it scores 0.565.

**Latency**, one request, single stream, in process:

| hardware | request | time |
|---|---|--:|
| RTX 5060 Ti, bf16 | ≈400-token state, 1 question (3 options) | 28.2 ms |
| RTX 5060 Ti, bf16 | ≈400-token state, 3 questions (8 options) | 34.0 ms |
| CPU, 6 threads, PyTorch fp32 | 384-token state, 3 questions (8 options) | 2.28 s |
| CPU, 6 threads, ONNX int8 | 384-token state, 3 questions (8 options) | 0.88 s |

GPU rows: median of 30 requests after warm-up, timing parsing, tokenisation, packing and the forward pass. CPU rows were measured on a machine with background load; the ratio between them (about 2.5x) is the stable part.

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
