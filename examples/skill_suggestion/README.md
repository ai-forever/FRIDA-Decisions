# Один интент из большого каталога

Сначала `ranking` по каталогу, потом `noul` только для топ-5: «клиент хочет именно это?». Если лучшая вероятность ниже порога, код не выбирает ничего. При почти равных вероятностях остаётся порядок ranking: int8 ONNX может поменять местами два ответа по 0.98.

По умолчанию пул — 40 интентов. Полный каталог из 243: `--full` у скрипта и `fast=False` в ноутбуке. Каталог — [`examples/data/intent_catalog.json`](../data/intent_catalog.json).

Паттерн TypeSafe: [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion).

```bash
python examples/skill_suggestion/skill_suggestion.py --device cpu
python examples/skill_suggestion/skill_suggestion.py --onnx
python examples/skill_suggestion/skill_suggestion.py --device cpu --full
```

Ноутбук: [skill_suggestion.ipynb](skill_suggestion.ipynb).
