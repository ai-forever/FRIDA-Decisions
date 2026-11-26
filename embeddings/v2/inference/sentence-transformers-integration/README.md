# Sentence Transformers integration

The converter constructs native `Transformer`, `Pooling`, `Dense`, `Normalize`
and `MultiVectorMask` modules and calls `SentenceTransformer.save_pretrained()`.
ST writes the module configs, version metadata, processor/tokenizer and weights.
Chat-call options use `processing_kwargs`, preserving the saved chat template.

Any PEFT adapter is merged into the full Qwen3.5 vision/language backbone before
saving. The resulting folder is the final model for ST, vLLM and SGLang. There
is no separate engine exporter. Engines select the custom architecture with
their command-line override, without changing the ST config. Only safetensors
files are gitignored.

```text
Transformer (materialized Qwen3.5 vision + language backbone)
  → LAST Pooling                          → sentence_embedding [2048]
  → bias-free Dense over token_embeddings → multivector_embeddings [tokens, 128]
  → Normalize(sentence_embedding)
  → Normalize(multivector_embeddings)
  → MultiVectorMask
```

Both heads share one backbone forward. ST casts the trained projection to the
backbone inference dtype during normal module initialization.

Use Sentence Transformers 6.3+:

```bash
uv sync --project sentence-transformers-integration
uv run --project sentence-transformers-integration joint-st-convert \
  /path/to/training-checkpoint /path/to/st-model --max-length 4096 --pool-at-eos
```

Output is never overwritten. Base weights must be cached or downloadable;
`--local-files-only` enforces a cached export. Set prefixes and text context
length to match training. The complete saved model needs no base download when
loaded. GPU engine serving and retrieval quality need separate validation.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("ai-forever/FRIDA-Embeddings-V2-2B")
items = model.encode_document(["Example document"], output_value=None)
for item in items:
    dense = item["sentence_embedding"]
    multivector = item["multivector_embeddings"][item["attention_mask"].bool()]
    print(dense.shape, multivector.shape)

# Uses the configured query prefix:
queries = model.encode_query(["Example query"], output_value=None)
```

`encode()` returns dense embeddings by default. `output_value=None` preserves
both fields. Use `attention_mask` to remove padded multivector rows. Both heads are
normalized by their own modules. Load this joint revision as
`SentenceTransformer`; converting it to `MultiVectorEncoder` can remove dense
pooling. No MaxSim scoring mask is silently added.

For batched joint inference with this package installed:

```python
from inference_plugins_sentence_transformers.encode import encode_joint

items = encode_joint(model, ["Document one.", "Document two."], role="document", batch_size=32)
dense, tokens = items[0]["dense"], items[0]["multivector"]
```

The helper runs both heads in one forward pass per batch and returns CPU tensors with padding removed.
It extracts only the embedding tensors, so scalar preprocessing metadata in Sentence Transformers 6.3
can be shared across the whole batch.

The default exported text limit is **4096 tokens**. Both tokenizer `max_length`
and `model_max_length` are saved with that value. ST derives its limit from the
saved tokenizer; engine context limits can be set independently for serving.
