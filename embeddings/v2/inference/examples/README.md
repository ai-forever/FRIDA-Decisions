# Примеры инференса

Запускайте команды из `embeddings/v2/inference`. Вместо `/models/frida` подставьте путь к полной модели в формате Sentence Transformers.

| Что проверить | Пример |
| --- | --- |
| Документы или запросы локально | [sentence_transformers_offline.py](sentence_transformers_offline.py) с `--role document` или `--role query` |
| vLLM на тексте | [vllm_offline.py](vllm_offline.py), [vllm_online.py](vllm_online.py) |
| Изображения через vLLM | [vllm_multimodal_offline.py](vllm_multimodal_offline.py), [vllm_multimodal_online.py](vllm_multimodal_online.py) |
| Форматы ответа vLLM | [vllm_binary_online.py](vllm_binary_online.py) |
| SGLang | [sglang_offline.py](sglang_offline.py), [sglang_online.py](sglang_online.py) |

```bash
uv run --project vllm-integration python examples/vllm_offline.py \
  --model /models/frida --texts 'Что такое семантический поиск?'
bash examples/test_vllm_container.sh --model /models/frida
bash examples/test_sglang_container.sh --model /models/frida
```

Запускайте каждый контейнерный тест в окружении соответствующего движка. vLLM принимает текст, ID токенов или сообщения. При передаче сырого текста префикс запроса или документа задаёт клиент. [Контракт HTTP и форматы ответов](../vllm-integration/README.md).
