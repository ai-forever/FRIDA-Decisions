"""Run the stock vLLM CLI with the installed plugins."""

from __future__ import annotations


def main() -> None:
    """Delegate to the stock CLI."""
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
