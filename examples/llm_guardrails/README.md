# Модерация сообщений

На каждое сообщение — набор `noul` (оскорбление, jailbreak, персональные данные, опасный запрос) и `score` тяжести. Пороги в коде выбирают pass, review или block. Все сообщения считаются одним `judge_batch`. Размеченные сообщения лежат в `DATA`.

Паттерн TypeSafe: [Guardrails for LLMs](https://docs.typesafe.ai/cookbooks/llm_guardrails).

```bash
python examples/llm_guardrails/llm_guardrails.py --device cpu
python examples/llm_guardrails/llm_guardrails.py --onnx
```

Ноутбук: [llm_guardrails.ipynb](llm_guardrails.ipynb).
