# Examples

Прикладные примеры FRIDA-Decisions. Модель отвечает на типизированные вопросы о тексте и возвращает вероятности; решение — порог, вес, маршрут — собирается в коде, а не в промпте. Соседние постановки задач: [cookbooks TypeSafe](https://docs.typesafe.ai/cookbooks) и [patterns](https://docs.typesafe.ai/patterns).

Каждый сценарий — папка: ноутбук, который открывается в Colab, скрипт с теми же вопросами и той же логикой. Данные лежат в коде (`DATA`), файлов рядом нет. Номера в ноутбуках — вывод выполненного прогона на Colab CPU, не иллюстрация.

Скрипт запускается так же, как [`quickstart.py`](quickstart.py):

```bash
python examples/<имя>/<имя>.py                 # torch, GPU если есть, иначе CPU
python examples/<имя>/<имя>.py --device cpu
python examples/<имя>/<имя>.py --onnx          # int8 ONNX на CPU
python examples/<имя>/<имя>.py --model path/to/exported/folder
```

`skill_suggestion` по умолчанию ранжирует пул из 40 интентов: `fast=True` в ноутбуке, `--full` у скрипта даёт все 243.

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
| [support_triage](support_triage/) | Маршрут обращения по confidence: команда, срочность, риск ухода | choice, score, noul | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/support_triage/support_triage.ipynb) | [`support_triage.py`](support_triage/support_triage.py) |
| [llm_guardrails](llm_guardrails/) | pass / review / block по вероятностям рисков и тяжести | noul, score | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/llm_guardrails/llm_guardrails.ipynb) | [`llm_guardrails.py`](llm_guardrails/llm_guardrails.py) |
| [rag_filtering](rag_filtering/) | BM25, затем ranking и noul: что отдать в контекст | ranking, noul | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/rag_filtering/rag_filtering.ipynb) | [`rag_filtering.py`](rag_filtering/rag_filtering.py) |
| [function_calling](function_calling/) | Выбор функции и аргументов из закрытых списков, числа — регуляркой | choice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/function_calling/function_calling.ipynb) | [`function_calling.py`](function_calling/function_calling.py) |
| [composite_scoring](composite_scoring/) | Несколько атомарных score, веса и порог в коде | score | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/composite_scoring/composite_scoring.ipynb) | [`composite_scoring.py`](composite_scoring/composite_scoring.py) |
