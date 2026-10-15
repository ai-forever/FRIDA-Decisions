# Выбор уже найденной строки

Регулярное выражение собирает телефоны, суммы или даты. `choice` выбирает строку, которая отвечает на вопрос, либо `none`. Код нормализует выбранный фрагмент и не выдумывает символы, которых в тексте не было.

Паттерн TypeSafe: [Pre-parsed value extraction](https://docs.typesafe.ai/cookbooks/pre_parsed_value_extraction_cookbook).

```bash
python examples/value_extraction/value_extraction.py --device cpu
python examples/value_extraction/value_extraction.py --onnx
```

Ноутбук: [value_extraction.ipynb](value_extraction.ipynb).
