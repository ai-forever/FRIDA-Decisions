# FRIDA Decisions inference

The Python distribution `frida-decisions` includes the FRIDA Decisions SDK and vLLM plugins. For a V2 checkpoint, install the serving dependencies in a dedicated environment with `python -m pip install '.[v2]'`. The V2 plugin targets vLLM 0.32.0 and Transformers 5.19.0.

Set `VLLM_PLUGINS=frida_decisions_v2` before starting vLLM. The plugin registers the V2 model and the IO processor used by `POST /pooling`. Follow the selected checkpoint's model card for the complete serving command and supported text and image inputs. The V1 plugin is registered separately as `frida_decisions`.

The V2 checkpoint also provides a standalone Hugging Face decision pipeline. Its model card describes the separate Transformers environment and inference examples.
