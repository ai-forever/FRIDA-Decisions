# Отбор пассажей для RAG

`rank_bm25` берёт топ короткого списка, `ranking` переставляет его, фильтр «margin минус среднее по списку» оставляет тех, кто выше списка. Отдельный `noul` на каждый фрагмент ловит случай, когда ответа нет вообще.

Паттерны TypeSafe: [Classifying RAG passages](https://docs.typesafe.ai/cookbooks/classifying_rag_passages) и [Re-ranking](https://docs.typesafe.ai/cookbooks/rerank_typesafe).

```bash
pip install rank_bm25
python examples/rag_filtering/rag_filtering.py --device cpu
python examples/rag_filtering/rag_filtering.py --onnx
```

Ноутбук: [rag_filtering.ipynb](rag_filtering.ipynb). Порог `3.5` взят из README репозитория и относится к спискам другой формы; на этом игрушечном наборе цифры только показывают приём.
