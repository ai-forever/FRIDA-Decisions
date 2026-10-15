# Один интент из большого каталога

Сначала `ranking` по каталогу, потом `noul` только для топ-5: «клиент хочет именно это?». Если лучшая вероятность ниже порога, код не выбирает ничего.

По умолчанию пул — 40 интентов, среди них все размеченные. Полный каталог из 243 интентов: `--full` у скрипта и `FAST = False` в ноутбуке. Каталог — [`examples/data/intent_catalog.json`](../data/intent_catalog.json), он уже в репозитории.

Паттерн TypeSafe: [Skill suggestion](https://docs.typesafe.ai/cookbooks/skill_suggestion).

```bash
python examples/skill_suggestion/skill_suggestion.py --device cpu
python examples/skill_suggestion/skill_suggestion.py --onnx
python examples/skill_suggestion/skill_suggestion.py --device cpu --full
```

Ноутбук: [skill_suggestion.ipynb](skill_suggestion.ipynb).
