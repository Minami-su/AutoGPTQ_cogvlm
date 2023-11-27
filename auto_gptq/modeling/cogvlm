from ._base import *


class CogVlmGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "CogVLMDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.vision_expert_query_key_value"],
        ["self_attn.vision_expert_dense"],
        ["self_attn.language_expert_query_key_value"],
        ["self_attn.language_expert_dense"],
        ["mlp.language_mlp.up_proj", "mlp.language_mlp.gate_proj"],
        ["mlp.language_mlp.down_proj"],
        ["mlp.vision_mlp.up_proj", "mlp.vision_mlp.gate_proj"],
        ["mlp.vision_mlp.down_proj"]
    ]


__all__ = ["CogVlmGPTQForCausalLM"]
