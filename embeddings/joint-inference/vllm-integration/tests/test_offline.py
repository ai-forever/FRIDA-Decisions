"""Combined output needs no mutable per-request selection state."""
from concurrent.futures import ThreadPoolExecutor


def test_offline_combined_requests_are_stateless(processor):
    def encode(index):
        params = processor.merge_pooling_params()
        assert params.task == 'token_embed'
        assert params.extra_kwargs == {'joint_embeddings': True}
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(executor.map(encode, range(12)))


def test_encode_joint_preserves_native_prompts_and_per_input_params(processor, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock

    import torch
    from inference_plugins_vllm.offline import encode_joint
    from vllm.pooling_params import PoolingParams

    llm = SimpleNamespace(
        model_config=SimpleNamespace(model='local-st', architecture='Qwen35JointEmbedding'),
        llm_engine=SimpleNamespace(vllm_config=object()),
        encode=Mock(),
    )
    monkeypatch.setattr('inference_plugins_vllm.offline.load_checkpoint', lambda config: processor)
    prompts = [{'prompt': '<image>', 'multi_modal_data': {'image': object()}}, [1, 2]]
    packed = [torch.arange(8.).reshape(4, 2), torch.arange(6.).reshape(3, 2)]
    llm.encode.return_value = [SimpleNamespace(outputs=SimpleNamespace(data=p)) for p in packed]
    params = [PoolingParams(extra_kwargs={'user': i}) for i in range(2)]
    result = encode_joint(llm, prompts, params, tokenization_kwargs={'add_special_tokens': False})
    args, kwargs = llm.encode.call_args
    assert args[0] is prompts and kwargs['pooling_task'] == 'token_embed'
    assert kwargs['tokenization_kwargs'] == {'add_special_tokens': False}
    for i, param in enumerate(kwargs['pooling_params']):
        assert param.task == 'token_embed' and param.extra_kwargs == {'user': i, 'joint_embeddings': True}
        assert params[i].task is None and params[i].extra_kwargs == {'user': i}
    assert result == [
        {'dense': [0, 1, 2, 3], 'multivector': [[4, 5], [6, 7]]},
        {'dense': [0, 1, 2, 3], 'multivector': [[4, 5]]},
    ]
