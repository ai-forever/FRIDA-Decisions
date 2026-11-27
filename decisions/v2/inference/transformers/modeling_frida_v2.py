"""Portable text and image decision scoring with the FRIDA V2 serving tensors."""
import torch
from torch import nn
from transformers import AutoConfig, AutoImageProcessor, AutoModel, PreTrainedModel
from transformers.decision import DecisionConfig
from transformers.decision.modeling_utils import DecisionMixin
from transformers.decision.template_utils import Question

from .configuration_frida_v2 import FridaV2Config
from .decision_confidence import confidence
from .decision_images import extract_images
from .decision_limits import check_request_length, prepare_scalar_inputs
from .decision_prefix import score_shared_state


class SpecialRows(nn.Module):
    def __init__(self, embedding, ids, scale):
        super().__init__()
        self.base = embedding
        self.register_buffer("ids", torch.tensor(ids))
        self.delta = nn.Parameter(torch.zeros(len(ids), embedding.embedding_dim, dtype=torch.float32))
        self.scale = scale

    @property
    def weight(self):
        return self.base.weight

    def forward(self, ids):
        original = self.base(ids)
        mask = (ids[..., None] == self.ids).to(self.delta.dtype)
        return (original.float() + mask @ (self.delta * self.scale)).to(original.dtype)


class FridaQuestion(Question):
    def answer(self, probabilities):
        answer = super().answer(probabilities)
        if self.type in ("choice", "score"):
            answer["confidence"] = confidence(probabilities)
        return answer


class ImageDecisions(DecisionConfig):
    def prepare_inputs(self, request, tokenizer):
        request, images = extract_images(request)
        prompts = prepare_scalar_inputs(self, request, tokenizer)
        if images:
            size = dict(self.processor.size)
            size["longest_edge"] = self.pixel_limit
            size["shortest_edge"] = min(size["shortest_edge"], self.pixel_limit)
            media = self.processor(images=images, size=size, return_tensors="pt")
            for inputs, _ in prompts:
                inputs.update(pixel_values=media.pixel_values, image_grid_thw=media.image_grid_thw)
        for inputs, questions in prompts:
            check_request_length(inputs, questions, self.processor.merge_size)
        return [(inputs, [FridaQuestion(**vars(q)) for q in questions]) for inputs, questions in prompts]


class FridaV2ForDecision(DecisionMixin, PreTrainedModel):
    config_class = FridaV2Config
    _supports_sdpa = True
    _keep_in_fp32_modules = ["head", "delta"]

    def __init__(self, config):
        super().__init__(config)
        base = dict(config.backbone_config)
        vl_config = AutoConfig.for_model(base.pop("model_type"), **base)
        vl = AutoModel.from_config(vl_config, attn_implementation="sdpa")
        self.backbone = vl.language_model
        self.visual = vl.visual
        self.__dict__["_vl"] = vl
        # Earlier exports store the trained rows as offsets from a frozen table.
        if getattr(config, "train_tokens", None):
            self.backbone.set_input_embeddings(SpecialRows(self.backbone.get_input_embeddings(),
                                                          config.train_tokens, config.special_rows_scale))
        self.head = nn.Linear(vl_config.text_config.hidden_size, 1, dtype=torch.float32)
        self.post_init()

    @classmethod
    def from_pretrained(cls, path, *args, config=None, **kwargs):
        if config is not None and type(config) is not cls.config_class:
            config = cls.config_class.from_dict(config.to_dict())
        hub = {k: kwargs[k] for k in ("revision", "cache_dir", "token", "local_files_only") if k in kwargs}
        decisions = ImageDecisions.from_pretrained(path, **hub)
        decisions.processor = AutoImageProcessor.from_pretrained(path, subfolder="base", **hub)
        model = super().from_pretrained(path, *args, config=config, decision_config=decisions, **kwargs)
        decisions.pixel_limit = model.config.max_pixels
        return model

    def forward(self, input_ids, marks, num_options, pixel_values=None, image_grid_thw=None,
                use_shared_state=None, **kwargs):
        bounds = [0, *marks[0].tolist()]
        pieces = [input_ids[0, a:b] for a, b in zip(bounds, bounds[1:])]
        state = pieces[0]
        features = None
        positions = None
        if pixel_values is not None:
            cfg = self._vl.config
            block = []
            for grid in image_grid_thw:
                pads = int(grid.prod()) // cfg.vision_config.spatial_merge_size ** 2
                block.extend([cfg.vision_start_token_id, *([cfg.image_token_id] * pads), cfg.vision_end_token_id])
            state = torch.cat((state[:1], state.new_tensor(block), state[1:]))
            positions, _ = self._vl.get_rope_index(state[None], image_grid_thw=image_grid_thw,
                                                    mm_token_type_ids=(state[None] == cfg.image_token_id).int())
            features = self._vl.get_image_features(pixel_values.to(self.visual.dtype), image_grid_thw,
                                                   return_dict=True).pooler_output
            features = torch.cat(features, dim=0)
        groups = []
        cursor = 1
        for count in num_options[0].tolist():
            if count < 1 or cursor + count >= len(pieces):
                raise ValueError("Unexpected decision template segmentation")
            groups.append((pieces[cursor], pieces[cursor+1:cursor+1+count]))
            cursor += count + 1
        if cursor != len(pieces) or not groups:
            raise ValueError("Unexpected decision template segmentation")
        if use_shared_state is None:
            # On both release checkpoints branching lost time on short prefixes.
            # Count the actual shared state, including merged patches and wrappers.
            use_shared_state = len(state) >= 1536
        if use_shared_state and not self.training and not torch.is_grad_enabled():
            embeds = None
            if features is not None:
                embeds = self.backbone.get_input_embeddings()(state[None])
                embeds[state[None] == cfg.image_token_id] = features.to(embeds.dtype)
            return score_shared_state(self.backbone, self.head, state, groups, embeds, positions)
        return torch.cat([self._score_independent(state, question, options, features, positions)
                          for question, options in groups])

    def _score_independent(self, state, question, options, features, positions):
        scores = []
        for option in options:
            ids = torch.cat((state, question, option))[None]
            if features is None:
                hidden = self.backbone(input_ids=ids, use_cache=False).last_hidden_state
            else:
                embeds = self.backbone.get_input_embeddings()(ids)
                embeds[ids == self._vl.config.image_token_id] = features.to(embeds.dtype)
                tail = torch.arange(ids.shape[1] - len(state), device=ids.device) + positions.max() + 1
                pos = torch.cat((positions, tail.expand(3, 1, -1)), dim=2)
                hidden = self.backbone(inputs_embeds=embeds, position_ids=pos, use_cache=False).last_hidden_state
            scores.append(self.head(hidden[0, -1].float()).reshape(()))
        return torch.stack(scores)
