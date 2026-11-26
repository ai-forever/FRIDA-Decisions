# vLLM integration

All tasks use native vLLM inputs. Select the output on `/pooling`:

| `task` | Output per input |
| --- | --- |
| `embed` | Dense vector in `response["data"][i]["data"]` |
| `token_embed` | Multivector matrix in `response["data"][i]["data"]` |
| `plugin` | Both named `dense` and `multivector` fields in `response["data"][i]` |

The `joint_embeddings` endpoint plugin uses vLLM’s standard pooling route and
native chat/completion rendering. It selects the requested head on each HTTP
request. Prefixes and chat formatting remain caller-controlled.

## Serve

```bash
python -m pip install --no-deps --no-build-isolation -e ./core -e ./vllm-integration

VLLM_PLUGINS=joint_embeddings vllm serve ai-forever/FRIDA-Embeddings-V2-2B \
  --runner pooling --served-model-name joint-model \
  --hf-overrides '{"architectures":["Qwen35JointEmbedding"]}' \
  --no-enable-prefix-caching \
  --max-model-len 32768
```

Use a vLLM build with the `vllm.endpoint_plugins` interface and route replacement support
(the local checkout used for verification is `b7186d137a`).
Include `joint_embeddings` in the server’s `VLLM_PLUGINS` allowlist to load the
model and HTTP endpoint. `joint-vllm` and `frida-joint-serve` invoke the stock CLI.
Use the Hub ID of the joint Sentence Transformers checkpoint or its local directory.
`--revision` selects the model revision for both the backbone and the projection.
Prefix caching must remain disabled because multivector output needs hidden states
for the full prompt. Native pooling supports chunked prefill.

To save the engine options in `serve.yaml`:

```yaml
runner: pooling
served-model-name: joint-model
hf-overrides:
  architectures: [Qwen35JointEmbedding]
disable-prefix-caching: true
max-model-len: 32768
```

```bash
VLLM_PLUGINS=joint_embeddings vllm serve ai-forever/FRIDA-Embeddings-V2-2B --config serve.yaml
```

## HTTP inputs

All tasks accept native `input` (text strings or token IDs) and `messages`:

```bash
curl -sS --fail-with-body http://localhost:8000/pooling \
  -H 'Content-Type: application/json' -d '{
    "model": "joint-model", "task": "plugin",
    "input": ["Example document", "Another document"]
  }'
```

Change `task` to `embed` or `token_embed` for one head. Text batches use native
completion-style preprocessing. For preformatted token IDs, set
`add_special_tokens: false`. Add any training prefixes and EOS tokens explicitly.

Images use native chat content parts, including URLs or base64 data URLs:

```json
{
  "model": "joint-model", "task": "plugin", "add_generation_prompt": false,
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    {"type": "text", "text": "Describe this page"}
  ]}]
}
```

Replace `...` with real image bytes encoded as base64. Native chat requests apply
the saved chat template. HTTP `messages` represents one conversation; submit
separate requests for separate conversations. Video URLs and model-specific
embedding content parts use the same native parser, subject to model support.
Qwen3.5 does not support audio input.

```bash
python examples/vllm_online.py --url http://localhost:8000 --model joint-model \
  --task plugin --texts 'Example document'
python examples/vllm_multimodal_online.py --url http://localhost:8000 \
  --model joint-model --image page.png --task token_embed
```

## Official plugin data envelope

Combined output also accepts the official `task="plugin"` / `data` contract.
For HTTP, `data` contains exactly one of `input` or `messages`. It is converted
to vLLM's native completion/chat request before context initialization, retaining
native validation, media/template checks and rendering. Explicit outer request
fields take precedence over fields inside `data`.

```bash
curl -sS --fail-with-body http://localhost:8000/pooling \
  -H 'Content-Type: application/json' -d '{
    "model": "joint-model", "task": "plugin", "encoding_format": "base64",
    "data": {"input": ["Example document", "Another document"]}
  }'
```

Image conversations work the same way:

```json
{
  "model": "joint-model", "task": "plugin", "encoding_format": "bytes",
  "data": {
    "messages": [{"role": "user", "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe this page"}
    ]}],
    "add_generation_prompt": false
  }
}
```

The output is identical to direct native input, including all encoding formats.
`plugin` always returns both heads; `embed_task` and `role` are rejected.

For the optional stock offline envelope, construct the LLM with
`io_processor_plugin="joint_embeddings"`. Its `data.input` contains native prompts (text, token IDs, prepared prompt
dictionaries, or batches). PIL images stay inside native `multi_modal_data`:

```python
result = llm.encode(
    {
        "data": {
            "input": [
                "Example document",
                {"prompt": formatted_prompt, "multi_modal_data": {"image": pil_image}},
            ]
        }
    },
    pooling_task="plugin",
)
```

Offline `data` accepts only `input`: pass tokenization/pooling options to
`LLM.encode` directly. HTTP `data.messages` uses the native chat renderer;
offline chat must be formatted into native prompts first. Neither form adds
role prefixes or custom preprocessing.

```bash
python examples/vllm_online.py --url http://localhost:8000 --model joint-model \
  --task plugin --data-envelope --encoding-format base64 --texts 'Example document'
python examples/vllm_multimodal_online.py --url http://localhost:8000 \
  --model joint-model --image page.png --task plugin --data-envelope
python examples/vllm_multimodal_offline.py --model /path/to/st-model \
  --image page.png --task plugin --data-envelope
```

## Offline inputs

Configure the offline LLM for native token embeddings. The helper returns one
dictionary of dense and multivector lists per prompt, in input order.

```python
from inference_plugins_vllm.offline import encode_joint
from vllm import LLM
from vllm.config import PoolerConfig

llm = LLM(
    model="/path/to/st-model",
    runner="pooling",
    hf_overrides={"architectures": ["Qwen35JointEmbedding"]},
    pooler_config=PoolerConfig(task="token_embed"),
    enable_prefix_caching=False,
)
heads = encode_joint(llm, ["Example document", {"prompt_token_ids": [1, 2, 3]}])
tokens = llm.encode(["Example document"], pooling_task="token_embed")
```

For dense-only inference, initialize with `PoolerConfig(task="embed")` and call
`llm.embed(...)`. Stock offline task selection applies per LLM instance.

Native prompt dictionaries support `prompt`, `prompt_token_ids`, `prompt_embeds`,
`multi_modal_data`, `multi_modal_uuids`, `mm_processor_kwargs`, `media_io_kwargs`,
and `cache_salt`. PIL images, NumPy images, tensors, video frames, and supported
precomputed features pass through `multi_modal_data` without custom conversion.
Native serialized byte prompts use the same offline path. Qwen3.5 cannot accept
encoder-decoder prompts.

### PIL images offline

Format the image placeholder explicitly using the checkpoint's template, then
attach the PIL object:

```python
from PIL import Image

image = Image.open("page.png").convert("RGB")
formatted = llm.get_tokenizer().apply_chat_template(
    [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this page"},
            ],
        }
    ],
    tokenize=False,
    add_generation_prompt=False,
)
result = encode_joint(llm, [{"prompt": formatted, "multi_modal_data": {"image": image}}])
```

Bare PIL objects and conversation lists are not prompt shortcuts. Use native
prompt dictionaries, and apply prefixes/templates explicitly as required by
training. Different native prompts can share an offline batch.

```bash
python examples/vllm_multimodal_offline.py --model /path/to/st-model \
  --image page.png --task plugin --max-model-len 32768
# Or --task embed / --task token_embed
```

Precomputed prompt embeddings require `--enable-prompt-embeds` /
`enable_prompt_embeds=True`. Multimodal embeddings require `--enable-mm-embeds`
and the model-specific feature layout. Unfiltered token pooling does not require
token IDs; token-ID filtering still does. vLLM validates enabled features,
shapes, context limits and supported modalities. Native single-head offline calls return embedding objects.

## Output encoding

All three `/pooling` tasks support top-level `encoding_format`: `float`,
`base64`, `bytes`, or `bytes_only`. The integration reuses vLLM's base64 encoder
and binary response builder. For large joint batches, send the explicit HTTP header
`Accept: application/vnd.frida.embeddings.v1` to put tensor offsets and shapes in
the response body. Read it with `inference_plugins_vllm.wire.decode(response.content)`. The dedicated token-embedding endpoint retains
its native float-only restriction.

### Base64 output

```bash
curl -sS --fail-with-body http://localhost:8000/pooling \
  -H 'Content-Type: application/json' -d '{
    "model": "joint-model", "task": "plugin", "input": ["Example document"],
    "encoding_format": "base64", "embed_dtype": "float32", "endianness": "little"
  }'
```

Each combined head carries reconstruction metadata:

```json
{
  "dense": {"data": "...", "shape": [2048], "embed_dtype": "float32", "endianness": "little"},
  "multivector": {"data": "...", "shape": [12, 128], "embed_dtype": "float32", "endianness": "little"}
}
```

Decode a head using the explicitly requested float32/little-endian layout:

```python
import base64
import numpy as np

head = response["data"][0]["multivector"]
vectors = np.frombuffer(base64.b64decode(head["data"], validate=True), dtype="<f4").reshape(
    head["shape"]
)
```

The native single-head response still carries a base64 string directly in its
`data` field. Dtypes and byte order follow vLLM's codec options.

### Bytes output

`bytes` returns `application/octet-stream` plus a JSON `metadata` response
header. Combined payloads concatenate dense then multivector bytes per input.
Each metadata item contains `index`, `input_index`, `head`, `shape`,
`embed_dtype`, `endianness`, `start`, and `end`. Offsets address the response body
directly. Original input tokens are counted once for usage. `bytes_only` returns
the same bytes without metadata; callers must know the shapes and offsets.

```bash
python examples/vllm_binary_online.py --url http://localhost:8000 --encoding-format bytes
python examples/vllm_binary_online.py --url http://localhost:8000 --encoding-format base64
python examples/vllm_multimodal_online.py --url http://localhost:8000 \
  --model joint-model --image page.png --task plugin --encoding-format base64
```

The last example uses base64 independently for image input and embedding output.

## Checkpoint and validation

Use the complete ST model produced by `joint-st-convert` directly. It builds
native ST modules, merges the backbone, and saves through the ST API. There is
no additional engine export. `JointCheckpoint` reads ST `modules.json`
and locates the bias-free Dense projection's `model.safetensors`, key
`linear.weight`. ST files own dimensions, normalization, prompts and text length.
`config.json` may add `version`, `append_eos`, `add_special_tokens` and
`skip_token_ids` to `joint_embeddings`; input formatting is caller-controlled.
The empty `MultiVectorMask` is supported; custom scoring filters require engine
support. There are no legacy projection sidecars.

The model inherits native `Qwen3_5ForConditionalGeneration` and its multimodal processor. Bare ST `language_model.*` weights are normalized
before the inherited native weight mapper. Native `DispatchPooler`, `LastPool`, `AllPool` and
embedding heads perform pooling; the plugin combines outputs from one backbone
pass. No ColQwen subclass is required.

CPU tests cover native input rendering, task validation, output codecs, pooling,
chunking, ST loading and tiny Transformers multimodal execution. Set
`VLLM_SOURCE_DIR` if the checkout is not the sibling `vllm` directory. Full GPU
validation remains in the container smoke scripts.
