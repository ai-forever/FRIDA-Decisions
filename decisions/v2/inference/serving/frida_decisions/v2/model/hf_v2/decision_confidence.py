"""Shared FRIDA confidence for serving and standalone checkpoints."""
import math


def confidence(probabilities):
    """Return one minus normalized entropy, with unit confidence for a single option."""
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1 - entropy / math.log(len(probabilities))))
