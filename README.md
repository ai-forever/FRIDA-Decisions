# FRIDA · Retrieval and structured decisions

<p align="center">
  <img src="assets/frida-mascot.jpg" width="520" alt="The FRIDA mascot holds up a bar chart and a check mark.">
</p>

**FRIDA-Embeddings** encodes queries and documents for semantic search. **FRIDA-Decisions** scores candidate answers for classification, rating and reranking.

[Try the Decisions demo](https://huggingface.co/spaces/ai-forever/FRIDA-Decisions) · [Explore the original FRIDA encoder](https://huggingface.co/ai-forever/FRIDA) · [Browse the cookbook](decisions/v1/examples/)

## Projects

| Project | Overview | Start here |
| --- | --- | --- |
| [**FRIDA-Embeddings**](embeddings/) | Semantic search over text. V2 adds image retrieval and can return both a dense vector and per-token vectors. | [V1 model card](https://huggingface.co/ai-forever/FRIDA) · [V2 inference guide](embeddings/v2/inference/) |
| [**FRIDA-Decisions**](decisions/) | A ticket router, a passage reranker, or a classifier whose labels you describe in plain language. The model scores the options you give it. | [V1 quickstart](decisions/v1/#quickstart) · [Passage reranking notebook](decisions/v1/examples/rag_filtering.ipynb) |
| [**razvilka**](benchmarks/razvilka/) | Benchmark for structured decisions with 735 Russian-language examples across 15 tasks. | [Dataset](https://huggingface.co/datasets/artemsnegirev/razvilka) · [Evaluation notebook](benchmarks/razvilka/run_razvilka.ipynb) |

## News

- **In preparation** · FRIDA-Embeddings V2 for text and image retrieval, and FRIDA-Decisions V2 in two sizes (1.8B and 28B-A2.5B).
- **6 October 2026** · FRIDA-Decisions 0.4.0 brought local inference to Apple Silicon with [MLX](decisions/v1/docs/backends.md#mlx-on-apple-silicon).

[MIT license](LICENSE)
