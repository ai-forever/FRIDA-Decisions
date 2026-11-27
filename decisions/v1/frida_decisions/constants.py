"""Names that are meant to be changed in exactly one place.

The product name, the default Hugging Face repository and the wire names of the
question types all live here. Everything else in the package refers to these
constants, so a rename is a one-line change.
"""
from __future__ import annotations

# The public product name. Used in responses, logs and docs strings.
PRODUCT_NAME = "FRIDA-Decisions"

# Default Hugging Face repository the weights are published under.
DEFAULT_REPO_ID = "ai-forever/FRIDA-Decisions"


class QuestionType:
    """Wire names of the four question types.

    The strings are part of the request format: a question declares
    `"type": QuestionType.CHOICE` and so on, and the answer repeats the type
    under the same key. Rename them here and the whole package follows.
    """

    CHOICE = "choice"      # pick one option out of several described options
    SCORE = "score"        # place the input on an ordered scale of levels
    YES_NO = "noul"        # a yes/no question, answered with a probability
    RANKING = "ranking"    # order candidate texts (passages, actions) best first

    ALL = (CHOICE, SCORE, YES_NO, RANKING)


# Labels of the two sides of a yes/no question (keys of its `criteria`).
YES_LABEL = "true"
NO_LABEL = "false"

# File names inside an exported model folder.
MODEL_WEIGHTS_FILE = "model.safetensors"
HEAD_WEIGHTS_FILE = "head.safetensors"
DECISIONS_CONFIG_FILE = "decisions_config.json"
ONNX_SUBDIR = "onnx"
ONNX_INT8_FILE = "model_int8_pertoken.onnx"
