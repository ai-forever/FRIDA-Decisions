"""Numerical dispatch tests; substitute only unavailable vLLM engine base classes."""

from __future__ import annotations

import ast
import dataclasses
from itertools import groupby
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from inference_plugins.contract import Contract
from inference_plugins.heads import JointHeads, unpack


def native_classes(relative, names, **extra):
    """Execute the actual checkout classes without importing CUDA engine packages."""
    checkout = Path(os.environ.get('VLLM_SOURCE_DIR', str(Path(__file__).resolve().parents[3] / 'vllm')))
    path = checkout / 'vllm' / relative
    if not path.exists():
        pytest.skip('Local vLLM checkout is required for native pooling tests')
    tree = ast.parse(path.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names]
    namespace = dict(F=torch.nn.functional, envs=SimpleNamespace(VLLM_BATCH_INVARIANT=False), __name__=__name__, dataclass=dataclasses.dataclass, dataclasses=dataclasses, groupby=groupby, torch=torch, nn=torch.nn, ABC=ABC, abstractmethod=abstractmethod, TokenPoolingMethodOutputItem=torch.Tensor | None,
                     contextmanager=contextmanager, field=dataclasses.field,
                     register_module_module_registration_hook=torch.nn.modules.module.register_module_module_registration_hook,
                     get_current_vllm_config=lambda: SimpleNamespace(scheduler_config=SimpleNamespace(enable_chunked_prefill=False, async_scheduling=False)))
    namespace.update(extra)
    code = 'from __future__ import annotations\n' + ast.unparse(ast.Module(body=selected, type_ignores=[]))
    exec(compile(code, str(path), 'exec'), namespace)
    return namespace


@pytest.fixture
def pooler_class(monkeypatch):
    update = native_classes('model_executor/layers/pooler/common.py', {'PoolingParamsUpdate'})['PoolingParamsUpdate']
    token = native_classes('model_executor/layers/pooler/tokwise/methods.py', {'TokenPoolingMethod', 'AllPool'}, PoolingParamsUpdate=update)
    seq = native_classes('model_executor/layers/pooler/seqwise/methods.py', {'SequencePoolingMethod', 'LastPool'})
    utils = native_classes('model_executor/models/utils.py', {'StageMissingLayer', 'no_init_weights'})
    base = native_classes('model_executor/layers/pooler/abstract.py', {'Pooler'}, PoolingParamsUpdate=update)['Pooler']
    dispatch = native_classes('model_executor/layers/pooler/special.py', {'DispatchPooler'}, Pooler=base)
    seq_heads = native_classes('model_executor/layers/pooler/seqwise/heads.py', {'SequencePoolerHead', 'EmbeddingPoolerHead'})
    tok_heads = native_classes('model_executor/layers/pooler/tokwise/heads.py', {'TokenPoolerHead', 'TokenEmbeddingPoolerHead'})
    activations = native_classes('model_executor/layers/pooler/activations.py', {'PoolerActivation', 'PoolerNormalize'})
    token_pooler = native_classes('model_executor/layers/pooler/tokwise/poolers.py', {'TokenPooler'},
        Pooler=base, PoolingParamsUpdate=update, TokenPoolingMethod=token['TokenPoolingMethod'],
        AllPool=token['AllPool'], TokenPoolerHead=tok_heads['TokenPoolerHead'],
        TokenEmbeddingPoolerHead=tok_heads['TokenEmbeddingPoolerHead'], POOLING_TASKS={'embed', 'token_embed'})


    class FakeHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(4, 4))

        def tie_weights(self, embedding):
            self.weight = embedding.weight
            return self

    class Language(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Embedding(8, 8)
            self.lm_head = FakeHead().tie_weights(self.model)
            self.logits_processor = FakeHead()

    class Backbone(torch.nn.Module):
        hf_to_vllm_mapper = object()
        def __init__(self, **kwargs):
            super().__init__()
            self.language_model = Language()
            self.visual = torch.nn.Linear(3, 8, bias=False)

        def load_weights(self, weights):
            return {'language_model.model.weight', 'visual.weight'}

    modules = {
        'vllm.model_executor.layers.pooler.abstract': {'Pooler': base},
        'vllm.model_executor.layers.pooler.activations': {'PoolerNormalize': activations['PoolerNormalize']},
        'vllm.model_executor.layers.pooler.tokwise.poolers': {'TokenPooler': token_pooler['TokenPooler']},
        'vllm.model_executor.layers.pooler.common': {'PoolingParamsUpdate': update},
        'vllm.model_executor.layers.pooler.seqwise.methods': {'LastPool': seq['LastPool']},
        'vllm.model_executor.layers.pooler.special': {'DispatchPooler': dispatch['DispatchPooler']},
        'vllm.model_executor.layers.pooler.seqwise.heads': {'EmbeddingPoolerHead': seq_heads['EmbeddingPoolerHead']},
        'vllm.model_executor.layers.pooler.tokwise.heads': {'TokenEmbeddingPoolerHead': tok_heads['TokenEmbeddingPoolerHead']},
        'vllm.model_executor.layers.pooler.tokwise.methods': {'AllPool': token['AllPool']},
        'vllm.model_executor.layers.logits_processor': {'LogitsProcessor': FakeHead},
        'vllm.model_executor.layers.vocab_parallel_embedding': {'ParallelLMHead': FakeHead},
        'vllm.model_executor.models.utils': {name: utils[name] for name in ('StageMissingLayer', 'no_init_weights')},
        'vllm.model_executor.models.qwen3_5': {'Qwen3_5ForConditionalGeneration': Backbone, 'Qwen3_5ProcessingInfo': object},
        'vllm.model_executor.models.qwen3_vl': {'Qwen3VLDummyInputsBuilder': object, 'Qwen3VLMultiModalProcessor': object},
        'vllm.multimodal': {'MULTIMODAL_REGISTRY': SimpleNamespace(register_processor=lambda *args, **kwargs: lambda cls: cls)},
    }
    for name, attributes in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__file__).resolve().parents[1] / 'src/inference_plugins_vllm/model.py'
    spec = importlib.util.spec_from_file_location('inference_plugins_vllm.joint_pooler_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def factory(heads):
        joint = module.JointPooler(module.TokenPooler(heads.contract, heads.projection), module.DensePooler(heads.contract))
        result = dispatch['DispatchPooler']({'embed': joint.embed_pooler, 'token_embed': joint})
        result.__dict__['reference'] = heads
        return result
    factory.model_class = module.Qwen35JointEmbedding
    return factory


def metadata(tasks, lengths):
    native = native_classes('v1/pool/metadata.py', {'PoolingCursor', 'PoolingStates', 'PoolingMetadata'})
    sizes = torch.tensor(lengths)
    ends = sizes.cumsum(0)
    cursor = native['PoolingCursor'](ends - sizes, ends - 1, sizes, sizes, sizes, False, [True] * len(tasks))
    return native['PoolingMetadata'](
        prompt_lens=sizes, prompt_token_ids=None,
        prompt_token_ids_cpu=torch.stack([torch.arange(max(lengths)) for _ in lengths]),
        pooling_params=[SimpleNamespace(task=task, dimensions=None, use_activation=None, extra_kwargs=None, late_chunking_params=None) for task in tasks],
        pooling_states=[native['PoolingStates']() for _ in tasks], pooling_cursor=cursor,
    )


def test_mixed_tasks_preserve_heads_and_order(pooler_class: type[torch.nn.Module]) -> None:
    torch.manual_seed(7)
    c = Contract(dense_dim=8, multivector_dim=2, skip_token_ids=(0,))
    weight = torch.randn(2, 8)
    hidden = torch.randn(9, 8)
    heads = JointHeads(c, weight)
    pooler = pooler_class(heads)
    m = metadata(["embed", "token_embed", "token_embed"], [2, 3, 4])
    m.pooling_params[-1].extra_kwargs = {"joint_embeddings": True}
    output = pooler(hidden, m)
    dense = torch.nn.functional.normalize(hidden[1], dim=-1)
    tokens = torch.nn.functional.normalize(hidden[2:5] @ weight.T, dim=-1)[1:]
    torch.testing.assert_close(output[0], dense)
    torch.testing.assert_close(output[1], tokens)
    joint = unpack(output[2], c)
    torch.testing.assert_close(
        torch.tensor(joint["dense"]), torch.nn.functional.normalize(hidden[-1], dim=-1)
    )
    torch.testing.assert_close(
        torch.tensor(joint["multivector"]),
        torch.nn.functional.normalize(hidden[5:] @ weight.T, dim=-1)[1:],
    )
    assert pooler.get_supported_tasks() == {"embed", "token_embed"}


def test_dense_never_executes_projection_or_reads_token_ids(
    pooler_class: type[torch.nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    heads = JointHeads(Contract(dense_dim=8, multivector_dim=2), torch.randn(2, 8))
    monkeypatch.setattr(
        heads.projection, "forward", Mock(side_effect=AssertionError("projection executed"))
    )
    m = metadata(["embed"], [3])
    m.get_prompt_token_ids_cpu = Mock(side_effect=AssertionError("token IDs requested"))
    output = pooler_class(heads)(torch.randn(3, 8), m)
    assert output[0].shape == (8,)


def test_tokens_never_compute_dense(
    pooler_class: type[torch.nn.Module], monkeypatch: pytest.MonkeyPatch
) -> None:
    heads = JointHeads(Contract(dense_dim=8, multivector_dim=2), torch.randn(2, 8))
    monkeypatch.setattr(heads, "dense", Mock(side_effect=AssertionError("dense head executed")))
    output = pooler_class(heads)(torch.randn(3, 8), metadata(["token_embed"], [3]))
    assert output[0].shape == (3, 2)


def test_explicit_normalization_and_invalid_dimensions(pooler_class: type[torch.nn.Module]) -> None:
    weight, hidden = torch.randn(2, 8), torch.randn(3, 8)
    pooler = pooler_class(JointHeads(Contract(dense_dim=8, multivector_dim=2), weight))
    m = metadata(["embed"], [3])
    m.pooling_params[0].use_activation = False
    torch.testing.assert_close(pooler(hidden, m)[0], hidden[-1])
    m.pooling_params[0].dimensions = 2
    with pytest.raises(ValueError, match="truncation"):
        pooler(hidden, m)


def test_native_joint_task_preserves_contract(pooler_class):
    torch.manual_seed(23)
    contract = Contract(dense_dim=8, multivector_dim=2, skip_token_ids=(0,), normalize_dense=False)
    pooler = pooler_class(JointHeads(contract, torch.randn(2, 8)))
    hidden = torch.randn(9, 8)
    native = metadata(["embed", "token_embed", "token_embed"], [2, 3, 4])
    native.pooling_params[-1].extra_kwargs = {"joint_embeddings": True}
    native.pooling_params[-1].use_activation = False
    actual = pooler(hidden, native)
    chunks = hidden.split([2, 3, 4])
    from inference_plugins.heads import pack
    expected = [pooler.reference.dense(chunks[0]), pooler.reference.multivector(chunks[1], torch.arange(3)),
                pack(*pooler.reference(chunks[2], torch.arange(4)), contract)]
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
    assert actual[-1].shape == (7, 2)  # Four dense rows and three retained tokens.


def test_joint_flag_rejects_dense_execution(pooler_class):
    pooler = pooler_class(JointHeads(Contract(dense_dim=8, multivector_dim=2), torch.randn(2, 8)))
    m = metadata(["embed"], [3])
    m.pooling_params[0].extra_kwargs = {"joint_embeddings": True}
    with pytest.raises(ValueError, match="token_embed"):
        pooler(torch.randn(3, 8), m)


def test_chunked_mixed_batch_matches_full_prompts_and_owns_buffers(pooler_class):
    torch.manual_seed(31)
    contract = Contract(dense_dim=8, multivector_dim=2, skip_token_ids=(0,), normalize_dense=False)
    heads = JointHeads(contract, torch.randn(2, 8))
    pooler = pooler_class(heads)
    pooler.poolers_by_task['token_embed'].token_pooler.pooling.enable_chunked_prefill = True
    pooler.poolers_by_task['token_embed'].token_pooler.pooling.clone_finished = True
    full = [torch.randn(3, 8), torch.randn(5, 8)]
    states = []
    for _ in full:
        cache = []
        states.append(SimpleNamespace(hidden_states_cache=cache, clean=cache.clear))

    def step(lengths, done):
        m = metadata(['embed', 'token_embed'], lengths)
        m.pooling_states = states
        m.pooling_params[1].extra_kwargs = {'joint_embeddings': True}
        m.prompt_lens = torch.tensor([3, 5])
        m.prompt_token_ids_cpu = torch.stack([torch.arange(5), torch.arange(5)])
        m.pooling_cursor.prompt_lens_cpu = m.prompt_lens
        m.pooling_cursor.seq_lens_cpu = m.prompt_lens if all(done) else torch.tensor([1, 2])
        m.pooling_cursor.finished_mask = done
        return m

    first = torch.cat([full[0][:1], full[1][:2]])
    assert pooler(first, step([1, 2], [False, False])) == [None, None]
    first.fill_(-999)  # Simulate the scheduler reusing the previous step's buffer.
    final = torch.cat([full[0][1:], full[1][2:]])
    dense, packed = pooler(final, step([2, 3], [True, True]))
    torch.testing.assert_close(dense, heads.dense(full[0]))
    from inference_plugins.heads import pack
    torch.testing.assert_close(packed, pack(*heads(full[1], torch.arange(5)), contract))
    saved_dense = dense.clone()
    final.zero_()
    torch.testing.assert_close(dense, saved_dense)
    assert all(not state.hidden_states_cache for state in states)




def test_legacy_execution_task_is_rejected(pooler_class):
    pooler = pooler_class(JointHeads(Contract(dense_dim=8, multivector_dim=2), torch.randn(2, 8)))
    with pytest.raises(ValueError, match='Unsupported task'):
        pooler(torch.randn(3, 8), metadata(['plugin'], [3]))


def test_dispatch_slices_noncontiguous_task_groups(pooler_class):
    torch.manual_seed(42)
    heads = JointHeads(Contract(dense_dim=8, multivector_dim=2), torch.randn(2, 8))
    pooler = pooler_class(heads)
    lengths = [2, 3, 4, 1]
    hidden = torch.randn(sum(lengths), 8)
    m = metadata(['token_embed', 'embed', 'token_embed', 'embed'], lengths)
    outputs = pooler(hidden, m)
    for i, part in enumerate(hidden.split(lengths)):
        expected = heads.dense(part) if m.tasks[i] == 'embed' else heads.multivector(part, torch.arange(lengths[i]))
        torch.testing.assert_close(outputs[i], expected)
    assert all(not m.pooling_states[i].hidden_states_cache for i in (1, 3))


def test_model_loads_st_projection_without_legacy_sidecar(pooler_class, tmp_path):
    import json
    import shutil
    from safetensors.torch import save_file
    source = Path(__file__).resolve().parents[2] / 'tests/fixtures/st'
    shutil.copytree(source, tmp_path, dirs_exist_ok=True, ignore=shutil.ignore_patterns('*.safetensors', 'tokenizer*'))
    (tmp_path / 'tokenizer_config.json').write_text(json.dumps({'model_max_length': 32768}))
    for folder, dimension in [('1_Pooling', 'embedding_dimension'), ('2_Dense', 'in_features')]:
        path = tmp_path / folder / 'config.json'
        config = json.loads(path.read_text())
        config[dimension] = 8
        if folder == '2_Dense':
            config['out_features'] = 2
        path.write_text(json.dumps(config))
    weight = torch.randn(2, 8)
    save_file({'linear.weight': weight}, tmp_path / '2_Dense/model.safetensors')
    config = SimpleNamespace(cache_config=SimpleNamespace(enable_prefix_caching=False),
        model_config=SimpleNamespace(model=str(tmp_path), hf_config=SimpleNamespace()))
    model = pooler_class.model_class(vllm_config=config)
    torch.testing.assert_close(model.joint_pooler.token_pooler.head.projector.weight, weight)
    assert model.pooler.get_supported_tasks() == {'embed', 'token_embed'}
    assert 'hf_to_vllm_mapper' not in type(model).__dict__
    assert set(dict(model.named_parameters())) == {'language_model.model.weight', 'visual.weight', 'joint_pooler.token_pooler.head.projector.weight'}
    assert model.language_model.lm_head.stage_name == 'output'
    assert model.language_model.logits_processor.stage_name == 'output'
    assert model.language_model.model.weight.device.type == 'cpu'
    assert model.load_weights([]) == {'language_model.model.weight', 'visual.weight', 'joint_pooler.token_pooler.head.projector.weight'}
    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match='prefix'):
        pooler_class.model_class(vllm_config=config)



@pytest.mark.parametrize('dense_norm,token_norm', [(False, False), (False, True), (True, False), (True, True)])
def test_native_joint_heads_preserve_independent_normalization(pooler_class, dense_norm, token_norm):
    from inference_plugins.heads import pack
    contract = Contract(dense_dim=8, multivector_dim=2, normalize_dense=dense_norm,
                        normalize_multivector=token_norm, skip_token_ids=(0, 1, 2))
    heads = JointHeads(contract, torch.randn(2, 8, dtype=torch.bfloat16))
    hidden = torch.randn(3, 8, dtype=torch.bfloat16)
    m = metadata(['token_embed'], [3])
    m.pooling_params[0].extra_kwargs = {'joint_embeddings': True}
    m.pooling_params[0].use_activation = False
    output = pooler_class(heads)(hidden, m)[0]
    torch.testing.assert_close(output, pack(*heads(hidden, torch.arange(3)), contract))
    assert output.shape == (4, 2)  # All token rows filtered, dense retained.


def test_engine_default_activation_does_not_add_missing_st_normalize(pooler_class):
    contract = Contract(dense_dim=8, multivector_dim=2, normalize_dense=False, normalize_multivector=False)
    heads = JointHeads(contract, torch.randn(2, 8))
    pooler = pooler_class(heads)
    m = metadata(['embed', 'token_embed'], [2, 3])
    for params in m.pooling_params:
        params.use_activation = True  # vLLM fills this default during verification.
    hidden = torch.randn(5, 8)
    dense, tokens = pooler(hidden, m)
    torch.testing.assert_close(dense, hidden[1])
    torch.testing.assert_close(tokens, hidden[2:] @ heads.projection.weight.T)


def test_unfiltered_multivector_and_joint_outputs_do_not_require_token_ids(pooler_class):
    c = Contract(dense_dim=8, multivector_dim=2)
    heads = JointHeads(c, torch.randn(2, 8))
    pooler = pooler_class(heads)
    m = metadata(['token_embed', 'token_embed'], [2, 3])
    m.prompt_token_ids_cpu = None
    m.pooling_params[-1].extra_kwargs = {'joint_embeddings': True}
    hidden = torch.randn(5, 8)
    assert not pooler.get_pooling_updates('token_embed').requires_token_ids
    outputs = pooler(hidden, m)
    expected = torch.nn.functional.normalize(hidden[:2] @ heads.projection.weight.T, dim=-1)
    torch.testing.assert_close(outputs[0], expected)
    result = unpack(outputs[1], c)
    assert len(result['multivector']) == 3
    torch.testing.assert_close(torch.tensor(result['dense']), torch.nn.functional.normalize(hidden[-1], dim=-1))


def test_materialized_st_backbone_names_reach_native_mapper(pooler_class, monkeypatch):
    cls = pooler_class.model_class
    seen = []
    def load_native(self, weights):
        seen.extend(weights)
        return {'backbone'}
    monkeypatch.setattr(cls.__mro__[1], 'load_weights', load_native)
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    tensor = torch.ones(1)
    loaded = model.load_weights([('language_model.layers.0.weight', tensor), ('visual.weight', tensor)])
    assert [name for name, _ in seen] == ['model.language_model.layers.0.weight', 'visual.weight']
    assert all(value is tensor for _, value in seen)
    assert loaded == {'backbone', 'joint_pooler.token_pooler.head.projector.weight'}


def test_dense_batch_with_mixed_normalization_flags(pooler_class):
    pooler = pooler_class(JointHeads(Contract(dense_dim=8, multivector_dim=2), torch.randn(2, 8)))
    hidden = torch.arange(1., 41.).reshape(5, 8)
    m = metadata(['embed', 'embed'], [2, 3])
    m.pooling_params[1].use_activation = False
    outputs = pooler(hidden, m)
    torch.testing.assert_close(outputs[0], torch.nn.functional.normalize(hidden[1], dim=-1))
    torch.testing.assert_close(outputs[1], hidden[-1])


@pytest.mark.parametrize("chunked", [True, False])
def test_token_projection_is_batched_when_eligible(pooler_class, monkeypatch, chunked):
    heads = JointHeads(Contract(dense_dim=8, multivector_dim=2), torch.randn(2, 8))
    project = Mock(wraps=heads.projection.forward)
    monkeypatch.setattr(heads.projection, 'forward', project)
    pooler = pooler_class(heads)
    pooler.poolers_by_task["token_embed"].token_pooler.pooling.enable_chunked_prefill = chunked
    hidden = torch.randn(5, 8)
    outputs = pooler(hidden, metadata(['token_embed', 'token_embed'], [2, 3]))
    assert project.call_count == 1
    assert project.call_args.args[0].shape == (5, 8)
    for result, part in zip(outputs, hidden.split([2, 3]), strict=True):
        torch.testing.assert_close(result, torch.nn.functional.normalize(part @ heads.projection.weight.T, dim=-1))
