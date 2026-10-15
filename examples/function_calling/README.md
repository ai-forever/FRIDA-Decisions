# Вызов обычных функций

`choice` выбирает функцию и аргументы из закрытых списков (город, тариф, срок). Номер заказа достаёт регулярное выражение. Если confidence функции низкий или обязательный аргумент не назван, код не вызывает функцию, а возвращает уточняющий вопрос.

Паттерн TypeSafe: [Function calling](https://docs.typesafe.ai/cookbooks/function_calling).

```bash
python examples/function_calling/function_calling.py --device cpu
python examples/function_calling/function_calling.py --onnx
```

Ноутбук: [function_calling.ipynb](function_calling.ipynb).
