# razvilka evaluation

Scorer and notebook for the [razvilka](https://huggingface.co/datasets/artemsnegirev/razvilka) benchmark: 735 Russian items, 15 tasks, four question types.

* `razvilka_eval.py` — single-file scorer, standard library only (`datasets` only to load from the Hub). Accuracy overall, per type and per task; argmax ties go to the lexicographically smallest option key.
* `run_razvilka.ipynb` — loads the data, runs FRIDA-Decisions, scores it next to a lexical baseline and chance, and shows an adapter template for any other model.

```python
import razvilka_eval as rz

items = rz.load()                       # from the Hub, or rz.load("test.jsonl")
report = rz.score(predictions, items)   # predictions: {item_id: answer or distribution}
print(rz.format_report(report))
```

Command line: `python razvilka_eval.py --predictions predictions.jsonl` with one `{"id": ..., "prediction": ...}` per line.
