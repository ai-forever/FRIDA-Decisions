# Иерархическая классификация

Дерево категорий товаров. На каждом уровне `choice` среди детей узлов, которые ещё в луче ширины 2. Если шаг в узел сделан неуверенно, метка откатывается к родителю.

Паттерн TypeSafe: [Hierarchical classification](https://docs.typesafe.ai/cookbooks/hierarchical_classification).

```bash
python examples/hierarchical_classification/hierarchical_classification.py --device cpu
python examples/hierarchical_classification/hierarchical_classification.py --onnx
```

Ноутбук: [hierarchical_classification.ipynb](hierarchical_classification.ipynb).
