# Examples

Прикладные примеры FRIDA-Decisions. Модель отвечает на типизированные вопросы о тексте и возвращает вероятности; решение — порог, вес, маршрут — собирается в коде, а не в промпте. Соседние постановки задач: [cookbooks TypeSafe](https://docs.typesafe.ai/cookbooks) и [patterns](https://docs.typesafe.ai/patterns).

Каждый сценарий — папка: ноутбук, который открывается в Colab, скрипт с теми же вопросами и той же логикой, и при необходимости небольшой русский `data.json`. Номера в ноутбуках — вывод выполненного прогона, не иллюстрация.

Скрипт запускается так же, как [`quickstart.py`](quickstart.py):

```bash
python examples/<имя>/<имя>.py                 # torch, GPU если есть, иначе CPU
python examples/<имя>/<имя>.py --device cpu
python examples/<имя>/<имя>.py --onnx          # int8 ONNX на CPU
python examples/<имя>/<имя>.py --model path/to/exported/folder
```

Тяжёлые ноутбуки держат `FAST = True`: на бесплатном Colab с CPU этого хватает, чтобы уложиться в несколько минут. Поставьте `FAST = False`, чтобы прогнать полный набор.

## Уже в репозитории

| Пример | Что показывает | Типы вопросов | Colab | Скрипт |
| --- | --- | --- | --- | --- |
| Quickstart | Первый запрос, четыре типа вопросов, порог по `ranking`, каталог из 243 интентов | choice, score, noul, ranking | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/notebooks/quickstart.ipynb) | [`quickstart.py`](quickstart.py) |
| Async | `await judge.judge`: несколько запросов одновременно попадают в один `judge_batch` | choice, noul | — | [`async_judge.py`](async_judge.py) |
| MLX | Тот же запрос на Apple Silicon, без torch | choice | — | [`mlx_quickstart.py`](mlx_quickstart.py) |
| vLLM | Клиент к серверу vLLM: один текст, повтор по тому же тексту, каталог, burst | choice | — | [`vllm_client.py`](vllm_client.py) |

## Сценарии

| Пример | Что показывает | Типы вопросов | Colab | Скрипт |
| --- | --- | --- | --- | --- |
| [composite_scoring](composite_scoring/) | Несколько атомарных score, веса и порог в коде | score | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/grok/examples-composite_scoring/examples/composite_scoring/composite_scoring.ipynb) | [`composite_scoring.py`](composite_scoring/composite_scoring.py) |
