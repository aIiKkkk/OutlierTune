import torch
import torch.nn as nn
from transformers.models.bloom.modeling_bloom import BloomBlock
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer

@torch.no_grad()
def symmetrization_ln_fcs(ln, fcs, act_scales):
    if not isinstance(fcs, list):
        fcs = [fcs]
    if not isinstance(act_scales, dict):
        pass
    else:
        act_scales = act_scales.get('input')
    assert isinstance(ln, nn.LayerNorm)
    for fc in fcs:
        assert isinstance(fc, nn.Linear)
        assert ln.weight.numel() == fc.in_features == act_scales.numel()

    device, dtype = fcs[0].weight.device, fcs[0].weight.dtype
    scales = act_scales.to(device=device, dtype=dtype)

    ln.bias.sub_(scales.view(ln.bias.shape))
    for fc in fcs:
        fc.bias.add_(torch.mm(scales.view(1,-1), fc.weight.t()).view(fc.bias.shape))

@torch.no_grad()
def symmetrization_lm(model, scales):
    for name, module in model.named_modules():
        if isinstance(module, OPTDecoderLayer):
            attn_ln = module.self_attn_layer_norm
            qkv = [module.self_attn.q_proj,
                   module.self_attn.k_proj, module.self_attn.v_proj]
            qkv_input_scales = scales[name + '.self_attn.q_proj']["input"]
            qkv_input_scales = qkv_input_scales / 512
            symmetrization_ln_fcs(attn_ln, qkv, qkv_input_scales)
            ffn_ln = module.final_layer_norm
            fc1 = module.fc1
            fc1_input_scales = scales[name + '.fc1']["input"]
            fc1_input_scales = fc1_input_scales / 512
            symmetrization_ln_fcs(ffn_ln, fc1, fc1_input_scales)
        elif isinstance(module, BloomBlock):
            attn_ln = module.input_layernorm
            qkv = module.self_attention.query_key_value
            qkv_input_scales = scales[name + '.self_attention.query_key_value']["input"]
            qkv_input_scales = qkv_input_scales / 512
            symmetrization_ln_fcs(attn_ln, qkv, qkv_input_scales)
            ffn_ln = module.post_attention_layernorm
            fc1 = module.mlp.dense_h_to_4h
            fc1_input_scales = scales[name + '.mlp.dense_h_to_4h']["input"]
            fc1_input_scales = fc1_input_scales / 512
            symmetrization_ln_fcs(ffn_ln, fc1, fc1_input_scales)

        elif isinstance(module, LlamaDecoderLayer):
            device, dtype = module.input_layernorm.weight.device, module.input_layernorm.weight.dtype

            input_gamma = (module.input_layernorm.weight).to(device).to(dtype)
            module.self_attn.q_proj.weight = module.self_attn.q_proj.weight.mul_(input_gamma)
            module.self_attn.k_proj.weight = module.self_attn.k_proj.weight.mul_(input_gamma)
            module.self_attn.v_proj.weight = module.self_attn.v_proj.weight.mul_(input_gamma)
            module.input_layernorm.weight =  module.input_layernorm.weight.div_(input_gamma)

            post_gamma = (module.post_attention_layernorm.weight).to(device).to(dtype)
            module.mlp.gate_proj.weight = module.mlp.gate_proj.weight.mul_(post_gamma)
            module.mlp.up_proj.weight = module.mlp.up_proj.weight.mul_(post_gamma)
            module.post_attention_layernorm.weight = module.post_attention_layernorm.weight.div_(post_gamma)

