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

**FRIDA-Decisions** makes structured decisions over Russian text in a single encoder pass: pick one of K options, place a text on an ordinal scale, answer yes / no, or rank candidates. The options are written as text inside the request, so a new label set is a new JSON, not a new training run. No generation, no output tokens, no parsing: every answer is one of the declared options, with its confidence.

It is built on [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (T5 encoder, 823M parameters) and runs on a consumer GPU.

* **On par with a commercial decision API on Russian.** 0.890 on the [razvilka](https://huggingface.co/datasets/artemsnegirev/razvilka) benchmark against 0.897 for TypeSafe Jev; the difference is not statistically significant (paired McNemar, p = 0.68). Ahead of every open model we measured.
* **Fast.** 32–45 ms per request on an RTX 5060 Ti (384-token text, 1–3 questions), in process.
* **Cheap to run.** 4.3 GiB of GPU memory at peak; an int8 ONNX build runs on CPU.
* **Packing.** All options of all questions share one sequence and the text is encoded once; with the state cache a follow-up question about the same text costs only its own tokens. A catalog of 243 intents is answered in 0.44 s, against 4.65 s for one sequence per option.

## Quickstart

```bash
pip install "frida-decisions[torch] @ git+https://github.com/ai-forever/frida-decisions"
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

CPU without PyTorch: `pip install "frida-decisions[onnx] @ git+..."` and `OnnxJudge.from_pretrained("ai-forever/FRIDA-Decisions")` — int8 weights and per-token int8 activations; its decisions agree with the fp32 model on 120 of 122 test decisions.

## Question types

| type | asks | returns |
|---|---|---|
| `choice` | which of K options is right | the option key and a distribution over all options |
| `score` | where the text sits on an ordinal scale | a level 0..K−1 and its distribution |
| `noul` | is a statement true | `p(true)` |
| `ranking` | order candidates for a query | the order and a margin per candidate |

Texts up to 512 tokens are the recommended range.

## Benchmarks

**razvilka** — 735 Russian items, 15 tasks (routing, intents, topic and sentiment classification, moderation, relevance ranking), all four question types, gold from published datasets. All models are run on the same requests and scored with the same script ([razvilka_eval.py](https://github.com/ai-forever/frida-decisions/tree/main/benchmarks/razvilka)).

| model | parameters | accuracy |
|---|--:|--:|
| TypeSafe Jev (commercial API) | — | 0.897 |
| **FRIDA-Decisions** | 823M | **0.890** |
| smolnikov/migom-2b | 1.9B | 0.853 |
| Mapika/decider-2b | 1.9B | 0.833 |
| smolnikov/kivok-0.3b | 0.3B | 0.619 |
| fastino/GLiNER2.5-multi-Decide | 287M | 0.576 |
| convaiinnovations/laya (multilingual) | — | 0.559 |
| KaLM-Reranker-V1-Nano | 786M | 0.521 |
| open-jev (DeBERTa-v3-large) | 437M | 0.490¹ |
| lexical baseline | — | 0.333 |
| chance | — | 0.257 |

¹ 140 of 735 texts exceed open-jev's input window and count as random answers; on the other 595 it scores 0.565.

**Packing**, RTX 5060 Ti, bf16, same model, one sequence per option vs packed with the state cache:

| scenario | candidates | tokens, naive / packed | time, naive / packed |
|---|--:|--:|--:|
| intent from a 243-intent support-bot catalog | 243 | 97,685 / 4,871 | 4.65 s / 0.44 s |
| full triage of a support email | 16 questions, 65 options | 23,487 / 1,853 | 1.10 s / 0.18 s |
| follow-up question about a cached document | 13 | 5,081 / 277 | 0.24 s / 25 ms |

## Training

* 1.42M examples (1.5M questions) over 151 question types in eight domains: relevance and RAG grounding, NLI and fact checking, moderation and safety, agents and tool choice, topic classification, sentiment and emotion, LLM request routing, intents and customer support.
* 71% Russian, 29% English; mostly human or naturally labelled data, about a quarter with LLM-generated text or model labels; instruction wordings augmented with paraphrases.
* LoRA rank 16 on the attention projections (q, k, v, o) of all 24 layers plus a scalar head, 4.7M trainable parameters; merged into the weights in this release. Listwise softmax for `choice` / `ranking`, pairwise BCE for `noul`.
* One epoch, 49.6 hours on a single RTX 5060 Ti (8 GB).

## Files

* `model.safetensors` — encoder weights, bf16
* `head.safetensors` — readout head, fp32
* `decisions_config.json` — packing limits and instruction suffixes
* `onnx/model_int8_pertoken.onnx` — CPU build

## License

MIT. Based on [ai-forever/FRIDA](https://huggingface.co/ai-forever/FRIDA) (MIT).

## Citation

TODO
