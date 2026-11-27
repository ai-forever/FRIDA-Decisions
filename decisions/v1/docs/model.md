# FRIDA-Decisions V1 architecture

**FRIDA-Decisions answers structured questions about a text — pick an option, place it on a scale, say yes or no, rank candidates — and returns probabilities, not generated text.** All questions about one text, and all their options, are scored together by the [FRIDA](https://huggingface.co/ai-forever/FRIDA) encoder in packed sequences — the text is never re-encoded per option, and there is no decoding, no output tokens, no answer parsing.

* **Four question types** in one request: `choice`, `score`, `noul` (yes/no) and `ranking`.
* **Packed, not repeated.** Questions and options sit next to one copy of the text; a typical request is a single encoder call, and a 243-intent catalog is still one request.
* **Order-independent by construction.** Options never attend to each other, so adding, removing or reordering options does not change any other option's margin.
* **Exact state cache.** The text is encoded once and its keys/values are reused across rows and repeated requests — the same margins, less compute.
* **GPU or CPU.** PyTorch on GPU or CPU, an int8 ONNX model for CPU that does not need torch, MLX on Apple Silicon, or a vLLM server.

Developed by УЭСМО, SberAI.

## How it works

The model is FRIDA (a T5 encoder) with a linear head: a candidate's margin is the head applied to the mean of its own token states. Instead of encoding `[text][question][option]` once per option, a request is packed into one sequence with a block mask:

```
 tokens:   [ state ......... ][ question 1 ][ opt A ][ opt B ][ opt C ][ question 2 ][ yes ][ no ]
 positions: 0 ............ S   S ...... S+q   S+q..    S+q..    S+q..   S ...... S+r   S+r.. S+r..

 state      -> sees the state only
 question   -> sees the state and itself
 option     -> sees the state, its own question and itself   (never another option)
```

Positions restart after each question, so every option sits exactly where it would sit if it were scored alone; the mask makes the rest identical too. Large catalogs are split into rows of at most 16 options / 1,024 tokens — exactly, since options are independent.

Because the state only ever attends to itself, its per-layer keys and values do not depend on the questions. The torch backend encodes the state once and runs the question/option rows against the cached keys/values. This is exact (float32 drift ≈ 6.7e-06), and it kicks in automatically when a request needs more than one row or the same text was seen recently.

For the 243-intent example in [`examples/data/intent_catalog.json`](../examples/data/intent_catalog.json) the encoder reads 97,685 tokens if every intent gets its own sequence, 9,836 (16 rows) when packed, and 4,871 with the state cache.

## Limits

* **State:** up to 512 tokens is the recommended range — the range the model was trained on. `Judge.from_pretrained(..., state_max=384)` sets where the state is cut (default 384); longer states are cut from the end, and `usage.state_truncated` tells you when that happened.
* **Instructions:** up to 96 tokens per question, including a short type-specific suffix the model was trained with.
* **Options:** up to 256 tokens each. The number of options is limited only by time and memory; rows hold up to 16 options / 1,024 tokens and are split automatically.
* **Language:** FRIDA is a Russian–English encoder, and the built-in instruction suffixes are in Russian.

The packing limits ship with the weights in `decisions_config.json` and are read by every backend.

## Model folder

```
config.json               T5 encoder config
model.safetensors         encoder weights (bfloat16), LoRA already merged
head.safetensors          decision head, float32: weight (1, 1536), bias (1,)
tokenizer.json, tokenizer_config.json
decisions_config.json     packing limits, instruction suffixes, relative-position settings
onnx/model_int8_pertoken.onnx   int8 graph for OnnxJudge (1.17 GiB)
```

`tools/export_model.py` builds this folder from a FRIDA LoRA adapter (merge `W += B·A·α/r`, no `peft` needed at inference), and `tools/export_onnx.py` adds the int8 ONNX graph.
