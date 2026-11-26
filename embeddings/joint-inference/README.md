# Инференс FRIDA-Embeddings-V2-2B

Для поиска можно получить вектор размерности 2048 на весь вход, токенные векторы размерности 128 или оба представления за один проход. Всем бэкендам подходит один и тот же каталог готовой модели в формате Sentence Transformers. Команды ниже запускаются из `embeddings/joint-inference`.

| Каталог | Что в нём |
| --- | --- |
| [core](core/) | Чтение конфигурации модели и весов проекции, общий формат результатов |
| [sentence-transformers-integration](sentence-transformers-integration/) | Локальный инференс и конвертер тренировочного чекпойнта |
| [vllm-integration](vllm-integration/) | Офлайн-инференс и HTTP-сервер с режимами `embed`, `token_embed`, `plugin` |
| [sglang-integration](sglang-integration/) | Модель, обработчик изображений и запуск через SGLang |

### Попробовать на документе

```bash
uv run --project sentence-transformers-integration python examples/sentence_transformers_offline.py \
  --role document --texts 'Как устроен семантический поиск?'
```

Для локальных весов передайте `--model /path/to/st-model`. Модель обучалась на примерах до 4096 токенов. Чтобы увеличить лимит на документ, передайте `processing_kwargs={"text": {"max_length": 32768, "truncation": True}}` в `encode_document`.

### Запустить сервер

Каждый бэкенд устанавливается в своё окружение. Для vLLM нужен интерфейс `vllm.endpoint_plugins` с заменой маршрута. Использован коммит `b7186d137addf271d4dd24c8d73c9985437baddc`, он закреплён для uv в конфигурации пакета. В окружении с готовой сборкой этого коммита установите `core` и `vllm-integration` и зависимости из их `pyproject.toml`, включая `orjson`. [Команды запуска и примеры HTTP](vllm-integration/README.md).

На `/pooling` параметр `task="plugin"` возвращает обе головы. Для JSON сервер сериализует массивы напрямую через orjson. Для больших батчей можно передать заголовок `Accept: application/vnd.frida.embeddings.v1`. В этом формате размеры и смещения хранятся в теле ответа. В Python результат читает `inference_plugins_vllm.wire.decode(response.content)`, поля каждого элемента называются `dense` и `colbert`.

При включённом chunked prefill готовые запросы одного шага без накопленного состояния используют общую проекцию батча. Для последовательностей, разделённых между шагами, используется накопление vLLM. Prefix caching при выдаче токенных векторов должен быть выключен.

[Примеры для текста и изображений](examples/README.md)
