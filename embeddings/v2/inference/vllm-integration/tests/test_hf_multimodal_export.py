"""Verify real tiny HF vision/language loading without downloads or a GPU."""

import pytest
import torch


def test_native_namespace_reloads_both_towers_and_runs_images(tmp_path):
    transformers = pytest.importorskip('transformers')
    config_class = getattr(transformers, 'Qwen3_5Config', None)
    if config_class is None:
        pytest.skip('Transformers Qwen3.5 support is required')
    torch.manual_seed(18)
    config = config_class(image_token_id=29, video_token_id=28, vision_start_token_id=30, vision_end_token_id=31,
        text_config=dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8, layer_types=['full_attention'],
            linear_num_key_heads=1, linear_num_value_heads=1, linear_key_head_dim=8, linear_value_head_dim=8,
            rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0, 'partial_rotary_factor': 1.0, 'mrope_section': [2, 1, 1]}),
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=16,
            patch_size=2, temporal_patch_size=1, spatial_merge_size=1, num_position_embeddings=16))
    original = transformers.AutoModel.from_config(config).eval()
    original.save_pretrained(tmp_path)
    loaded, info = transformers.AutoModel.from_pretrained(tmp_path, output_loading_info=True)
    loaded.eval()
    assert not info['missing_keys'] and not info['unexpected_keys']
    assert hasattr(loaded, 'visual') and hasattr(loaded, 'language_model')
    ids = torch.tensor([[1, 30, 29, 29, 29, 29, 31, 2]])
    inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids), image_grid_thw=torch.tensor([[1, 2, 2]]),
                  mm_token_type_ids=torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0]]))
    pixels = torch.randn(4, 12)
    with torch.inference_mode():
        expected = original(**inputs, pixel_values=pixels).last_hidden_state
        actual = loaded(**inputs, pixel_values=pixels).last_hidden_state
        changed = loaded(**inputs, pixel_values=-pixels).last_hidden_state
    assert actual.shape == (1, 8, 16) and torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, changed)
