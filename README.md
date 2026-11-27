# FRIDA · Retrieval and structured decisions

<p align="center">
  <img src="assets/frida-mascot.jpg" width="520" alt="The FRIDA mascot holds up a bar chart and a check mark.">
</p>

**FRIDA-Embeddings** encodes queries and documents for semantic search. **FRIDA-Decisions** scores candidate answers for classification, rating and reranking.

[Sber AI R&D showcase](https://mlrnd.ru/) · [Decisions demo](https://huggingface.co/spaces/ai-forever/FRIDA-Decisions) · [Cookbook](decisions/v1/examples/)

## Models

- [**FRIDA**](https://huggingface.co/ai-forever/FRIDA) (2025) — the original Russian–English text embedding model for semantic search. [Habr article (RU)](https://habr.com/ru/companies/sberdevices/articles/909924/)
- [**FRIDA-Decisions**](https://huggingface.co/ai-forever/FRIDA-Decisions) (October 2026) — scores choices, ratings, yes/no questions and rankings over text. [Quickstart](decisions/v1/#quickstart) · [Habr article (RU)](https://habr.com/ru/companies/sberbank/articles/1092056/)

### V2 (in preparation)

- [**FRIDA-Embeddings-V2-2B**](embeddings/v2/inference/) — text and image retrieval with dense and token-level embeddings.
- [**FRIDA-Decisions-V2-1.8B**](decisions/) — a compact model for structured decisions over text and images.
- [**FRIDA-Decisions-V2-28B-A2.5B**](decisions/) — a mixture-of-experts model for the same tasks, with 2.5B active parameters. An FP8 variant is also in preparation.

## Benchmarks

[**razvilka**](benchmarks/razvilka/) was released alongside the original FRIDA-Decisions in October 2026. It evaluates structured decisions on 735 Russian-language examples across 15 tasks. [Dataset](https://huggingface.co/datasets/artemsnegirev/razvilka) · [Evaluation notebook](benchmarks/razvilka/run_razvilka.ipynb)

## News

- **6 October 2026** · FRIDA-Decisions 0.4.0 brought local inference to Apple Silicon with [MLX](decisions/v1/docs/backends.md#mlx-on-apple-silicon).

[MIT license](LICENSE)
