# Сопоставление карточек

Две карточки одним запросом, обычным текстом: `score` «один и тот же товар» и два `noul` (бренд совпадает, артикул совпадает). Решение принимает score: от 2.5 — same, ниже 1.5 — different, между ними — review. Yes/no печатаются рядом и не отменяют score.

Паттерн TypeSafe: [Knowledge graph entity alignment](https://docs.typesafe.ai/cookbooks/entity_alignment).

```bash
python examples/entity_alignment/entity_alignment.py --device cpu
python examples/entity_alignment/entity_alignment.py --onnx
```

Ноутбук: [entity_alignment.ipynb](entity_alignment.ipynb).
