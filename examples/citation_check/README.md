# Проверка цитаты по источнику

`choice` с тремя исходами: источник подтверждает утверждение, противоречит ему или не содержит данных. Низкая confidence уходит человеку, а не в автоматический вердикт.

Паттерн TypeSafe: [Double-checking citations](https://docs.typesafe.ai/cookbooks/citation_check).

```bash
python examples/citation_check/citation_check.py --device cpu
python examples/citation_check/citation_check.py --onnx
```

Ноутбук: [citation_check.ipynb](citation_check.ipynb).
