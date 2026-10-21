# Examples

FRIDA-Decisions answers typed questions about a text — `choice`, `score`, `noul` (yes/no), `ranking` — and returns probabilities. The decision itself, the threshold, the weight, the route, is ordinary code. That is what these notebooks are about: each one takes a task that is usually handed to an LLM prompt, and keeps the judgement in code where it can be measured and changed.

Every notebook opens in Colab, installs the package and runs end to end on a CPU runtime. The data lives in the notebook, in a `DATA` list, and is synthetic. Set `FD_MODEL` to a local exported folder to run against your own copy instead of the Hub.

The numbers printed in the notebooks were recorded on one machine: an AMD Ryzen 5 7500F, 6 cores, torch on 6 threads, float32, no GPU. That is torch float32 on a CPU, the slowest path the package offers: the guardrail cell takes 14.3 s there and 0.45 s on an RTX 5060 Ti in bfloat16, and `OnnxJudge` is about 2.5x faster than torch float32 on the same CPU (see the repository README). The decisions do not change with the path; the last digit of a probability can.

## Scenarios

| Notebook | What it teaches | Question types | Colab |
| --- | --- | --- | --- |
| [Route support tickets by confidence](support_triage.ipynb) | A confidence gate: route what the model is sure about, hand the rest to a person | `choice`, `score`, `noul` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/support_triage.ipynb) |
| [Screen messages: risks, severity, and rules in code](guardrails.ipynb) | Several yes/no risks and a severity scale; the action is a rule in code | `noul`, `score` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/guardrails.ipynb) |
| [Classify down a tree, and stop when confidence drops](hierarchical_classification.ipynb) | Walk down a category tree and fall back to the parent when a step is not confident | `choice` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/hierarchical_classification.ipynb) |
| [Check a claim against the source](citation_check.ipynb) | Supported, contradicted, or not in the source — with a verdict a person can check | `choice` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/citation_check.ipynb) |
| [Filter passages before they reach the answering model](rag_filtering.ipynb) | Re-rank a retriever's shortlist, and drop it when nothing in it answers | `ranking`, `noul` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/rag_filtering.ipynb) |
| [Suggest one intent from a large catalog](skill_suggestion.ipynb) | Rank a 243-intent catalog, then re-check the top few and abstain if none fits | `ranking`, `noul` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/skill_suggestion.ipynb) |
| [Call an ordinary function](function_calling.ipynb) | Pick the function and its closed arguments in one request; numbers come from a regex | `choice` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/function_calling.ipynb) |
| [Pick a span the regex already found](value_extraction.ipynb) | A regex proposes the spans, the model picks one of them or `none` | `choice` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/value_extraction.ipynb) |
| [Align two product cards](entity_alignment.ipynb) | Are these two cards the same product? A score decides, two yes/no checks explain | `score`, `noul` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/entity_alignment.ipynb) |
| [Combine atomic scores with your own weights](composite_scoring.ipynb) | Score each factor on its own, then keep the weights and the cutoff in code | `score` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/composite_scoring.ipynb) |
| [Use the answers as features](text_features.ipynb) | Answers as columns: a few probabilities and a score feed an ordinary classifier | `noul`, `score` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/text_features.ipynb) |
| [Ask everything in one request](parallel_questions.ipynb) | A dozen questions in one encoder pass against one question per call | `choice`, `score`, `noul` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/examples/parallel_questions.ipynb) |

`parallel_questions` also ships as a script, [`parallel_questions.py`](parallel_questions.py): it is a measurement you re-run, not a text you read once.

## Start here, and serving

| | | |
| --- | --- | --- |
| [Quickstart notebook](../notebooks/quickstart.ipynb) | The first request, the four question types, a threshold on `ranking`, a 243-intent catalog | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ai-forever/FRIDA-Decisions/blob/main/notebooks/quickstart.ipynb) |
| [`quickstart.py`](quickstart.py) | The same, as a script | |
| [`async_judge.py`](async_judge.py) | `await judge.judge`: concurrent requests collected into one `judge_batch` | |
| [`mlx_quickstart.py`](mlx_quickstart.py) | The same request on Apple Silicon, without torch | |
| [`vllm_client.py`](vllm_client.py) | A client for the vLLM server: one text, a repeat of it, a catalog, a burst | |

## Adding one

Open a pull request. A scenario is one notebook in this folder, English prose with the data in the notebook, a row in the table above, and output from a run you actually did. If it needs a package beyond the model, install it in the notebook's first cell.
