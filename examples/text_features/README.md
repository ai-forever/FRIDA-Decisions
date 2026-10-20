# Признаки для табличной модели

По каждому отзыву — три `noul` и один `score`. Из них `LogisticRegression` учится отличать отзывы, которые стоит поднять человеку. Разбиение на обучение и проверку записано в `DATA`. Это игрушечный набор: рядом печатается точность большинства.

Нужен `scikit-learn`. Упрощённый вариант паттерна TypeSafe [Autoresearch feature discovery](https://docs.typesafe.ai/cookbooks/autoresearch_feature_discovery).

```bash
pip install scikit-learn
python examples/text_features/text_features.py --device cpu
python examples/text_features/text_features.py --onnx
```

Ноутбук: [text_features.ipynb](text_features.ipynb).
