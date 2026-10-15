# Фильтр сообщений чата

На каждое сообщение несколько `noul`. Код по порогам выбирает hide, blur, collapse, highlight или show, в этом порядке. Сообщения уходят одновременно через `AsyncJudge`, и совпавшие по времени запросы считаются одним батчем.

Паттерн TypeSafe: [Twitch chat filter](https://docs.typesafe.ai/cookbooks/twitch_chat_filter).

```bash
python examples/chat_filter/chat_filter.py --device cpu
python examples/chat_filter/chat_filter.py --onnx
```

Ноутбук: [chat_filter.ipynb](chat_filter.ipynb).
