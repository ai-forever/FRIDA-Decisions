"""Exercise batched extraction when ST carries scalar metadata such as the task name."""

import torch
from inference_plugins_sentence_transformers.encode import encode_joint


class Model(torch.nn.Module):
    prompts = {"query": "search: "}
    default_prompt_name = None
    device = torch.device("cpu")

    def preprocess(self, inputs, *, prompt, task, processing_kwargs):
        assert prompt == "search: " and task == "query"
        assert processing_kwargs == {"text": {"max_length": 16}}
        return {"lengths": torch.tensor([len(item) for item in inputs])}

    def forward(self, features, *, task):
        lengths = features["lengths"]
        mask = torch.arange(int(lengths.max())) < lengths[:, None]
        return {
            "task": task,
            "sentence_embedding": lengths[:, None].float(),
            "multivector_embeddings": mask[:, :, None].float(),
            "attention_mask": mask,
        }


def test_batched_joint_uses_only_embedding_fields():
    docs = ["x" * n for n in (4, 1, 6, 2, 3, 7, 5, 8, 9)]
    items = encode_joint(Model(), docs, role="query", batch_size=8, processing_kwargs={"text": {"max_length": 16}})
    assert len(items) == len(docs)
    for doc, item in zip(docs, items, strict=True):
        assert item["dense"].item() == len(doc)
        assert item["multivector"].shape == (len(doc), 1)
