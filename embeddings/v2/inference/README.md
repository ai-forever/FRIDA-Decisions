# FRIDA-Embeddings-V2-2B inference

For each input, the model can return a 2048-dimensional dense embedding, a 128-dimensional vector per token, or both in one forward pass. All three backends read the same Sentence Transformers checkpoint. Use `add_generation_prompt=True` for queries and documents and keep the complete chat suffix, including the newline characters after `</think>`. The dense vector comes from the final token. Run the commands below from `embeddings/v2/inference`.

| Package | Purpose |
| --- | --- |
| [core](core/) | Load the model configuration and projection weights, and read the embedding outputs. |
| [sentence-transformers-integration](sentence-transformers-integration/) | Run inference locally or convert a training checkpoint. |
| [vllm-integration](vllm-integration/) | Run offline inference or serve the `embed`, `token_embed`, and `plugin` tasks over HTTP. |
| [sglang-integration](sglang-integration/) | Load and serve the model and its image processor with SGLang. |

## Embed a document

```bash
uv run --project sentence-transformers-integration python examples/sentence_transformers_offline.py \
  --role document --texts 'Semantic search retrieves documents by meaning.'
```

To use local weights, pass `--model /path/to/st-model`. The model was trained on examples up to 4096 tokens. To raise the document limit, pass `processing_kwargs={"text": {"max_length": 32768, "truncation": True}}` to `encode_document`.

## Serve with vLLM

Install each backend in its own environment. The vLLM integration uses the `vllm.endpoint_plugins` interface to replace the pooling route. The package configuration pins vLLM commit `b7186d137addf271d4dd24c8d73c9985437baddc` for uv. If your environment already has a build of that commit, install `core`, `vllm-integration`, and the dependencies listed in their `pyproject.toml` files, including `orjson`. [Server setup and HTTP examples](vllm-integration/README.md).

On `/pooling`, request `task="plugin"` to get both heads. JSON responses serialize the arrays directly with orjson. For large batches, send `Accept: application/vnd.frida.embeddings.v1` to receive binary data with shapes and offsets in the response body. In Python, call `inference_plugins_vllm.wire.decode(response.content)`. Each item contains `dense` and `colbert`.

With chunked prefill enabled, requests that finish in one step with no cached hidden states can share a batched token projection. Requests spanning multiple steps use vLLM's pooling cache. Disable prefix caching when serving token embeddings.

[Text and image examples](examples/README.md)
