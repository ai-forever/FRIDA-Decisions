# Inference examples

Run these commands from `embeddings/v2/inference`. Replace `/models/frida` with the path to the full Sentence Transformers checkpoint.

| Use case | Example |
| --- | --- |
| Embed documents or queries locally | [sentence_transformers_offline.py](sentence_transformers_offline.py) with `--role document` or `--role query` |
| Embed text with vLLM | [vllm_offline.py](vllm_offline.py), [vllm_online.py](vllm_online.py) |
| Embed images with vLLM | [vllm_multimodal_offline.py](vllm_multimodal_offline.py), [vllm_multimodal_online.py](vllm_multimodal_online.py) |
| Decode vLLM responses | [vllm_binary_online.py](vllm_binary_online.py) |
| Run SGLang | [sglang_offline.py](sglang_offline.py), [sglang_online.py](sglang_online.py) |

```bash
uv run --project vllm-integration python examples/vllm_offline.py \
  --model /models/frida --texts 'What is semantic search?'
bash examples/test_vllm_container.sh --model /models/frida
bash examples/test_sglang_container.sh --model /models/frida
```

Run each container test in the environment for its engine. vLLM accepts text, token IDs, or messages. When passing raw text, include the appropriate query or document prefix. [HTTP inputs and response formats](../vllm-integration/README.md).
