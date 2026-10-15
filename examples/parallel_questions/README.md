# Много вопросов за один проход

Двенадцать вопросов к одному тексту: одним запросом и по запросу на вопрос. Скрипт печатает время, `usage` и `token_counts`, и проверяет, что решения совпали.

Паттерны TypeSafe: [Parallel questions](https://docs.typesafe.ai/cookbooks/parallel_questions) и [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out).

```bash
python examples/parallel_questions/parallel_questions.py --device cpu
python examples/parallel_questions/parallel_questions.py --onnx
```

Ноутбук: [parallel_questions.ipynb](parallel_questions.ipynb).
