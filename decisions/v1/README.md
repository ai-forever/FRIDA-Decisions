# FRIDA-Decisions V1

The Python package `frida-decisions` scores structured questions about Russian and English text with the original FRIDA encoder.

[Model weights](https://huggingface.co/ai-forever/FRIDA-Decisions) · [Interactive demo](https://huggingface.co/spaces/ai-forever/FRIDA-Decisions) · [Article in Russian](https://habr.com/ru/companies/sberbank/articles/1092056/)

## Install

```bash
# PyTorch backend (GPU or CPU)
pip install "frida-decisions[torch] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.4.0"
# int8 ONNX backend for CPU, without torch
pip install "frida-decisions[onnx] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.4.0"
# MLX backend for Apple Silicon, without torch
pip install "frida-decisions[mlx] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.4.0"
# vLLM server (GPU, Linux)
pip install "frida-decisions[vllm] @ git+https://github.com/ai-forever/FRIDA-Decisions@v0.4.0"
```

The core package needs only `numpy`, `tokenizers`, `safetensors` and `huggingface_hub`; each backend comes with its extra (`[torch]`, `[onnx]`, `[mlx]`, `[vllm]`). Python 3.10+. The weights are downloaded from the Hugging Face Hub on first use.

To install the code from a local clone, run `python -m pip install -e "./decisions/v1[torch]"` from the FRIDA repository root. Choose the extra for your backend.

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

More in [`examples/quickstart.py`](examples/quickstart.py) and the notebook [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) — run it in Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/decisions/v1/notebooks/quickstart.ipynb)

Worked scenarios (routing, guardrails, retrieval, tool choice, and others) are indexed in [`examples/README.md`](examples/README.md).

## Documentation

- [Request and response formats](docs/requests.md) and [ranking thresholds](docs/requests.md#filtering-ranking-candidates-by-a-threshold)
- [Model architecture, packing and input limits](docs/model.md)
- [Inference backends and the async API](docs/backends.md)
- [Evaluation results, model export and tests](docs/evaluation.md)
- [Razvilka benchmark](../../benchmarks/razvilka/)

## License

MIT — see [LICENSE](LICENSE).
The base model [FRIDA](https://huggingface.co/ai-forever/FRIDA) is © ai-forever and released under the MIT License.
