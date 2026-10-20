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
| [parallel_questions](parallel_questions/) | Много вопросов одним запросом против отдельного запроса на каждый | choice, score, noul | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/parallel_questions/parallel_questions.ipynb) | [`parallel_questions.py`](parallel_questions/parallel_questions.py) |
| [hierarchical_classification](hierarchical_classification/) | Beam search по дереву категорий, откат к родителю при низкой confidence | choice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/hierarchical_classification/hierarchical_classification.ipynb) | [`hierarchical_classification.py`](hierarchical_classification/hierarchical_classification.py) |
| [skill_suggestion](skill_suggestion/) | Топ-k по ranking на каталоге интентов, перепроверка noul | ranking, noul | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/skill_suggestion/skill_suggestion.ipynb) | [`skill_suggestion.py`](skill_suggestion/skill_suggestion.py) |
| [citation_check](citation_check/) | Подтверждает ли фрагмент утверждение, или данных не хватает | choice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/citation_check/citation_check.ipynb) | [`citation_check.py`](citation_check/citation_check.py) |
| [value_extraction](value_extraction/) | Регулярка находит строки, choice выбирает нужную | choice | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/value_extraction/value_extraction.ipynb) | [`value_extraction.py`](value_extraction/value_extraction.py) |
| [entity_alignment](entity_alignment/) | Один ли это товар: score решает, noul по бренду и артикулу рядом | score, noul | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/entity_alignment/entity_alignment.ipynb) | [`entity_alignment.py`](entity_alignment/entity_alignment.py) |
| [chat_filter](chat_filter/) | Правила чата в коде, сообщения параллельно через AsyncJudge | noul | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/chat_filter/chat_filter.ipynb) | [`chat_filter.py`](chat_filter/chat_filter.py) |
| [text_features](text_features/) | Ответы noul и score как признаки логистической регрессии | noul, score | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/text_features/text_features.ipynb) | [`text_features.py`](text_features/text_features.py) |
