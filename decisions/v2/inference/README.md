# FRIDA-Decisions V2 inference

The `frida-decisions` 0.5.0 distribution includes the V1 SDK and the V1 and V2 vLLM plugins.

Install the V2 serving dependencies from the repository root with `python -m pip install ".[v2]"`. You can also install from [serving/](serving/) with the same command. The V2 plugin targets vLLM 0.32.0 and Transformers 5.19.0.

Set `VLLM_PLUGINS=frida_decisions_v2` for V2. See the [1.8B model card](https://huggingface.co/ai-forever/FRIDA-Decisions-V2-1.8B) or [28B-A2.5B model card](https://huggingface.co/ai-forever/FRIDA-Decisions-V2-28B-A2.5B) for the launch command and HTTP example.

Both checkpoints also provide a standalone decision pipeline using Transformers 5.24.1. The checkpoint integration source is in [transformers/](transformers/).
