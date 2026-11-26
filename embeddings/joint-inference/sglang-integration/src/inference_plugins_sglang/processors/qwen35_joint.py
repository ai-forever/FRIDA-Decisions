"""Register the standard Qwen vision processor for our pooling-only model."""

from typing import ClassVar

from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

from inference_plugins_sglang.models.qwen35_joint import Qwen35JointEmbedding


class JointImageProcessor(QwenVLImageProcessor):
    """Use native image/video preprocessing without introducing another image policy."""

    models: ClassVar = [Qwen35JointEmbedding]
