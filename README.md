# FRIDA · Retrieval and structured decisions

<p align="center">
  <img src="assets/frida-mascot.jpg" width="520" alt="The FRIDA mascot holds up a bar chart and a check mark.">
</p>

**FRIDA-Embeddings** encodes queries and documents for semantic search. **FRIDA-Decisions** scores candidate answers for classification, rating and reranking.

[Sber AI R&D showcase](https://mlrnd.ru/) · [Decisions demo](https://huggingface.co/spaces/ai-forever/FRIDA-Decisions) · [Cookbook](decisions/v1/examples/)

## Models

### V2

- [**FRIDA-Embeddings-V2-2B**](https://huggingface.co/ai-forever/FRIDA-Embeddings-V2-2B) — text and image retrieval with dense and token-level embeddings. [Inference guide](embeddings/v2/inference/)
- [**FRIDA-Decisions-V2-1.8B**](https://huggingface.co/ai-forever/FRIDA-Decisions-V2-1.8B) — a compact model for structured decisions over text and images.
- [**FRIDA-Decisions-V2-28B-A2.5B**](https://huggingface.co/ai-forever/FRIDA-Decisions-V2-28B-A2.5B) — a mixture-of-experts model for the same tasks, with 2.5B active parameters. [Inference guide](decisions/v2/inference/)

### V1

- [**FRIDA-Decisions**](https://huggingface.co/ai-forever/FRIDA-Decisions) (October 2026) — scores choices, ratings, yes/no questions and rankings over text. [Quickstart](decisions/v1/#quickstart) · [Habr article (RU)](https://habr.com/ru/companies/sberbank/articles/1092056/)
- [**FRIDA**](https://huggingface.co/ai-forever/FRIDA) (2025) — the original Russian–English text embedding model for semantic search. [Habr article (RU)](https://habr.com/ru/companies/sberdevices/articles/909924/)

## Benchmarks

- [**razvilka**](benchmarks/razvilka/) was released alongside the original FRIDA-Decisions in October 2026. It evaluates structured decisions on 735 Russian-language examples across 15 tasks. [Dataset](https://huggingface.co/datasets/artemsnegirev/razvilka) · [Evaluation notebook](benchmarks/razvilka/run_razvilka.ipynb)
- [**ruMTEB**](https://huggingface.co/collections/ai-forever/ru-mteb) evaluates Russian-language text embeddings across seven task categories, including retrieval, classification and clustering.
  Snegirev et al. (NAACL 2025), [*The Russian-focused embedders’ exploration: ruMTEB benchmark and Russian embedding model design*](https://aclanthology.org/2025.naacl-long.12/). [BibTeX](https://aclanthology.org/2025.naacl-long.12.bib)

## News

- **FRIDA V2** adds multimodal retrieval with Embeddings 2B and structured decisions over text and images with Decisions 1.8B and 28B-A2.5B. The Decisions package is now at version 0.5.0.
- **6 October 2026** · FRIDA-Decisions 0.4.0 brought local inference to Apple Silicon with [MLX](decisions/v1/docs/backends.md#mlx-on-apple-silicon).

[MIT license](LICENSE)
