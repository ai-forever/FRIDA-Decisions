"""ST projection and contract loading needs no installed vLLM or ST runtime."""
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from inference_plugins.contract import Contract
from inference_plugins.checkpoint import JointCheckpoint


@pytest.fixture
def checkpoint(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
    source = Path(__file__).resolve().parents[2] / 'tests/fixtures/st'
    modules = json.loads((source / 'modules.json').read_text())
    for module in modules[1:]:
        config = json.loads((source / module['path'] / 'config.json').read_text())
        module['path'] = 'renamed_' + module['name']
        if module['type'].endswith('.Dense'):
            config.update(in_features=8, out_features=2)
            projection = torch.randn(2, 8)
        elif module['type'].endswith('.Pooling'):
            config['embedding_dimension'] = 8
        write(tmp_path / module['path'] / 'config.json', config)
    write(tmp_path / 'modules.json', modules)
    write(tmp_path / 'sentence_bert_config.json', {'module_output_name': 'token_embeddings'})
    write(tmp_path / 'tokenizer_config.json', {'model_max_length': 123})
    write(tmp_path / 'config_sentence_transformers.json', {'prompts': {'query': 'Q: ', 'document': 'D: '}})
    save_file({'linear.weight': projection}, tmp_path / modules[2]['path'] / 'model.safetensors')
    return tmp_path, projection, modules


def test_st_layout_uses_module_paths_and_saved_settings(checkpoint):
    root, projection, _ = checkpoint
    contract = JointCheckpoint.load(root).contract
    assert contract == Contract(dense_dim=8, multivector_dim=2, max_length=123, query_prefix='Q: ', document_prefix='D: ')
    torch.testing.assert_close(JointCheckpoint.load(root).load_projection(), projection)
    settings = {'skip_token_ids': [3], 'append_eos': True}
    (root / 'config.json').write_text(json.dumps({'joint_embeddings': settings}))
    assert JointCheckpoint.load(root).contract.skip_token_ids == (3,)
    settings['normalize_dense'] = False
    (root / 'config.json').write_text(json.dumps({'joint_embeddings': settings}))
    with pytest.raises(ValueError, match='only'):
        JointCheckpoint.load(root)


@pytest.mark.parametrize('commit_hash', [None, 'a' * 40])
def test_hub_checkpoint_uses_backbone_revision_and_exact_projection(checkpoint, tmp_path, monkeypatch, commit_hash):
    from fnmatch import fnmatch
    from shutil import copyfile
    from types import SimpleNamespace

    from inference_plugins_vllm import checkpoint as loader

    source, projection, modules = checkpoint
    (source / 'config.json').write_text('{}')
    cache = tmp_path / 'cache'
    commit = 'a' * 40
    snapshot = cache / 'models--owner--joint' / 'snapshots' / commit
    model = SimpleNamespace(model='owner/joint', hf_config=SimpleNamespace(_commit_hash=commit_hash),
                            revision='release', hf_token='test-token')
    config = SimpleNamespace(model_config=model, load_config=SimpleNamespace(download_dir=str(cache)))
    head = modules[2]['path'] + '/model.safetensors'
    downloads = []

    def download(repo, filename, *, revision, token, cache_dir):
        assert (repo, token, cache_dir) == ('owner/joint', 'test-token', str(cache))
        assert revision == ((commit_hash or 'release') if filename == 'config.json' else commit)
        downloads.append(filename)
        target = snapshot / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source / filename, target)
        return str(target)

    def download_metadata(repo, *, revision, token, cache_dir, allow_patterns):
        assert (repo, revision, token, cache_dir) == ('owner/joint', commit, 'test-token', str(cache))
        assert all('safetensors' not in pattern for pattern in allow_patterns)
        for path in [*source.glob('*.json'), *source.glob('*/config.json')]:
            relative = path.relative_to(source)
            if any(fnmatch(str(relative), pattern) for pattern in allow_patterns):
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                copyfile(path, target)
        return str(snapshot)

    monkeypatch.setattr(loader, 'hf_hub_download', download)
    monkeypatch.setattr(loader, 'snapshot_download', download_metadata)
    assert loader.load_checkpoint(config).contract == JointCheckpoint.load(source).contract
    assert downloads == ['config.json']
    assert not (snapshot / head).exists()
    torch.testing.assert_close(loader.load_checkpoint(config, with_projection=True).load_projection(), projection)
    assert downloads == ['config.json', 'config.json', head]


def test_normalization_is_read_per_output(checkpoint):
    root, _, modules = checkpoint
    (root / 'modules.json').write_text(json.dumps(modules[:3] + modules[4:]))
    contract = JointCheckpoint.load(root).contract
    assert not contract.normalize_dense and contract.normalize_multivector


def test_projection_and_pipeline_validation(checkpoint):
    root, _, modules = checkpoint
    contract = JointCheckpoint.load(root).contract
    save_file({'weight': torch.randn(2, 8)}, root / modules[2]['path'] / 'model.safetensors')
    with pytest.raises(ValueError, match='linear.weight'):
        JointCheckpoint.load(root).load_projection()
    save_file({'linear.weight': torch.randn(3, 8)}, root / modules[2]['path'] / 'model.safetensors')
    with pytest.raises(ValueError, match='shape'):
        JointCheckpoint.load(root).load_projection()
    modules[2]['path'] = '../outside'
    (root / 'modules.json').write_text(json.dumps(modules))
    with pytest.raises(ValueError, match='inside'):
        JointCheckpoint.load(root).contract


def test_projection_requires_st_modules_even_with_legacy_sidecar(tmp_path):
    contract = Contract(dense_dim=8, multivector_dim=2)
    (tmp_path / 'heads').mkdir()
    save_file({'weight': torch.randn(2, 8)}, tmp_path / 'heads/multivector_projection.safetensors')
    with pytest.raises(FileNotFoundError, match='modules.json'):
        JointCheckpoint.load(tmp_path).load_projection()






def test_contract_text_format_follows_st_message_modality(checkpoint):
    root, _, _ = checkpoint
    (root / 'sentence_bert_config.json').write_text(json.dumps({'max_seq_length': 123, 'modality_config': {'message': {'method': 'forward'}}, 'processing_kwargs': {'chat_template': {'add_generation_prompt': True}}}))
    contract = JointCheckpoint.load(root).contract
    assert contract.use_chat_template
    class Tokenizer:
        eos_token_id = 99
        def apply_chat_template(self, conversation, **kwargs):
            assert kwargs == {'tokenize': False, 'add_generation_prompt': True}
            return 'CHAT:' + conversation[0]['content'][0]['text']
        def __call__(self, texts, **kwargs):
            assert texts == ['CHAT:Q: search']
            assert kwargs == {'add_special_tokens': False, 'truncation': True, 'max_length': 123}
            return {'input_ids': [[1, 2, 3]]}
    assert contract.tokenize(Tokenizer(), ['search'], 'query') == [[1, 2, 3]]




@pytest.mark.parametrize("settings", [{"skiplist_words": ["."]}, {"keep_only_token_ids": [42]}])
def test_engine_rejects_unsupported_scoring_masks(checkpoint, settings):
    root, _, modules = checkpoint
    mask = modules[-1]
    assert mask['type'].endswith('.MultiVectorMask')
    (root / mask['path'] / 'config.json').write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='unfiltered MultiVectorMask'):
        JointCheckpoint.load(root)
