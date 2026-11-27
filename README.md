# ![FRIDA — retrieval and structured decisions](assets/frida.svg)

A customer writes, *“I paid once. Why are there two charges on my card?”* **FRIDA-Embeddings** can find the help articles about duplicate payments. **FRIDA-Decisions** can rank those passages by how well they answer the question, or route the message straight to billing.

[Try the Decisions demo](https://huggingface.co/spaces/ai-forever/FRIDA-Decisions) · [Explore the original FRIDA encoder](https://huggingface.co/ai-forever/FRIDA) · [Browse the cookbook](decisions/v1/examples/)

## Models

| Project | What you can build | Start here |
| --- | --- | --- |
| [**FRIDA-Embeddings**](embeddings/) | Semantic search over text. V2 adds image retrieval and can return both a dense vector and per-token vectors. | [V1 model card](https://huggingface.co/ai-forever/FRIDA) · [V2 inference guide](embeddings/v2/inference/) |
| [**FRIDA-Decisions**](decisions/) | A ticket router, a passage reranker, or a classifier whose labels you describe in plain language. The model scores the options you give it. | [V1 quickstart](decisions/v1/#quickstart) · [Passage reranking notebook](decisions/v1/examples/rag_filtering.ipynb) |

## News

- **In preparation** · FRIDA-Embeddings V2 for text and image retrieval, and FRIDA-Decisions V2 in two sizes (1.8B and 28B-A2.5B).
- **6 October 2026** · FRIDA-Decisions 0.4.0 brought local inference to Apple Silicon with [MLX](decisions/v1/docs/backends.md#mlx-on-apple-silicon).

[Full changelog](CHANGELOG.md) · [Razvilka benchmark](benchmarks/razvilka/) · [MIT license](LICENSE)
