# Составная оценка

Три отдельных `score` — стек, ясность письма, тон. Веса и порог приглашения заданы в коде: вторая схема весов переставляет тех же кандидатов, не меняя вопросы.

Паттерн TypeSafe: [Composite scoring](https://docs.typesafe.ai/patterns/composite-scoring).

```bash
python examples/composite_scoring/composite_scoring.py --device cpu
python examples/composite_scoring/composite_scoring.py --onnx
```

Ноутбук: [composite_scoring.ipynb](composite_scoring.ipynb).
