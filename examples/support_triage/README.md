# Маршрутизация обращений

Один запрос на тикет: команда (`choice`), срочность (`score`), раздражение и угроза ухода (`noul`). Код отправляет тикет в команду, только если confidence выше порога, иначе — оператору. Размеченные тикеты лежат в `DATA` в том же файле.

Паттерны TypeSafe: [Intent routing](https://docs.typesafe.ai/patterns/intent-routing) и [Confidence-gated routing](https://docs.typesafe.ai/patterns/confidence-routing).

```bash
python examples/support_triage/support_triage.py --device cpu
python examples/support_triage/support_triage.py --onnx
```

Ноутбук: [support_triage.ipynb](support_triage.ipynb).
