"""vLLM backend: FRIDA-Decisions as a vLLM pooling model with its own attention.

Installed with `pip install "frida-decisions[vllm]"`; vLLM finds it through two
entry points (pyproject.toml), so there is nothing to import:

* `vllm.general_plugins: frida_decisions -> register` -- the model class
  `FridaDecisionsModel` and its attention backend, in every vLLM process;
* `vllm.io_processor_plugins: frida_decisions -> io_processor` -- the request
  front end of `POST /pooling` (`server.DecisionsIO`).

Modules: `rows` (a packed row as marker-delimited token ids, and back; no
torch), `kernel` (the arithmetic; torch, no vLLM -- the CPU tests run it),
`attention`, `model`, `pooler`, `server` (the vLLM-facing parts). vLLM is
imported only by the last four, so the rest is testable without it.

Tested with vLLM 0.29.0. The backend relies on vLLM internals (attention
backends, poolers) that change between releases; other versions are untested,
and `register` says so in the server log.
"""

ARCH = "FridaDecisionsModel"
TESTED_VLLM = "0.29.0"


def register():
    import vllm
    from vllm import ModelRegistry
    from vllm.logger import init_logger
    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    if vllm.__version__ != TESTED_VLLM:
        init_logger(f"vllm.plugins.{__name__}").warning(
            "frida-decisions is tested with vLLM %s only; running %s, untested: check the "
            "margins against Judge before relying on them.", TESTED_VLLM, vllm.__version__)
    if ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(ARCH, "frida_decisions.vllm_backend.model:FridaDecisionsModel")
    # The layers pass the backend class directly; registering it under CUSTOM as
    # well keeps any vLLM code that resolves `layer.backend` by enum working.
    register_backend(AttentionBackendEnum.CUSTOM,
                     "frida_decisions.vllm_backend.attention.FridaAttentionBackend")


def io_processor() -> str:
    return "frida_decisions.vllm_backend.server.DecisionsIO"
