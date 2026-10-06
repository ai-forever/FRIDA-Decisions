"""Run a native MLX request on Apple Silicon."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys

REQUEST = {
    'state': 'I want to keep my phone number while switching operators.',
    'questions': {'intent': {
        'type': 'choice',
        'instructions': "What is the customer's intent?",
        'criteria': {'port': 'transfer an existing phone number',
                     'new': 'get a completely new phone number'},
    }},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='ai-forever/FRIDA-Decisions')
    parser.add_argument('--revision', default=None)
    parser.add_argument('--dtype', choices=['float32', 'bfloat16'], default='float32')
    parser.add_argument('--assert-no-torch', action='store_true')
    args = parser.parse_args()
    if args.assert_no_torch:
        assert importlib.util.find_spec('torch') is None, 'PyTorch must not be installed'
    import mlx.core as mx
    from frida_decisions import MlxJudge
    judge = MlxJudge.from_pretrained(args.model, revision=args.revision, dtype=getattr(mx, args.dtype))
    response = judge(REQUEST)
    # Exercise explicit state encoding and automatic reuse.
    judge.margins_cached(REQUEST)
    repeated = judge(REQUEST)
    assert repeated['usage']['state_cache'] == 'hit'
    assert judge.state_cache.hits == 1 and judge.state_cache.misses == 1
    assert all(isinstance(t, mx.array) for entry in judge.state_cache._items.values()
               for t in entry[0] + entry[1])
    assert response['usage']['backend'] == 'mlx'
    assert response['answers']['intent']['choice'] == 'port'
    assert not any(n == 'torch' or n.startswith('torch.') for n in sys.modules)
    assert 'frida_decisions.torch_backend' not in sys.modules
    print(json.dumps(response, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
