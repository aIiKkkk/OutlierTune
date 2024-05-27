import torch
from torch import nn
from typing import Optional, Tuple
from fake_quant_bloom import W8A8Linear_QKVFc1, W8A8Linear_OFc2
import torch.nn.functional as F
from layer_norm import LayerNorm

class QuantOPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        org_module: nn.Module,
        embed_dim: int,
        num_heads: int,
        weight_quant,
        act_quant,
        quantize_bmm_input,
        act_scales,
        i,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = 0.0
        self.head_dim = embed_dim // num_heads
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.quantize_bmm_input = quantize_bmm_input
        self.act_scales = act_scales
        self.i = i
        # 在这一部分对缩放进行分类
        self.qkv_input_scales = self.act_scales[f"model.decoder.layers.{self.i}.self_attn.q_proj"]["input"]
        self.q_output_scales = self.act_scales[f"model.decoder.layers.{self.i}.self_attn.q_proj"]["output"]
        self.k_output_scales = self.act_scales[f"model.decoder.layers.{self.i}.self_attn.k_proj"]["output"]
        self.v_output_scales = self.act_scales[f"model.decoder.layers.{self.i}.self_attn.v_proj"]["output"]
        self.o_input_scales = self.act_scales[f"model.decoder.layers.{self.i}.self_attn.out_proj"]["input"]

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = True

        self.k_proj = W8A8Linear_QKVFc1.from_float(org_module.k_proj, input_scales = self.qkv_input_scales, output_scales = self.q_output_scales,
                      weight_quant=self.weight_quant, act_quant=self.act_quant, quantize_output=self.quantize_bmm_input)
        self.v_proj = W8A8Linear_QKVFc1.from_float(org_module.v_proj, input_scales = self.qkv_input_scales, output_scales = self.k_output_scales,
                      weight_quant=self.weight_quant, act_quant=self.act_quant, quantize_output=self.quantize_bmm_input)
        self.q_proj = W8A8Linear_QKVFc1.from_float(org_module.q_proj, input_scales = self.qkv_input_scales, output_scales = self.v_output_scales,
                      weight_quant=self.weight_quant, act_quant=self.act_quant, quantize_output=self.quantize_bmm_input)
        self.out_proj = W8A8Linear_OFc2.from_float(org_module.out_proj, input_scales = self.o_input_scales,
                      weight_quant=self.weight_quant, act_quant='per_token')

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()



    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        is_cross_attention = key_value_states is not None

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling
        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
            print(key_states.shape)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
            print(key_states.shape)
        else:
            # self_attention
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

            key_states = self._shape(key_states, -1, bsz)
            value_states = self._shape(value_states, -1, bsz)

        if self.is_decoder:
            past_key_value = (key_states, value_states)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is {layer_head_mask.size()}"
                )
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )
        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value


class QuantOPTDecoderLayer(nn.Module):
    def __init__(
        self,
        ori_layer,
        embed_dim,
        num_heads,
        weight_quant,
        act_quant,
        quantize_bmm_input,
        config,
        act_scales,
        i
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.quantize_bmm_input = quantize_bmm_input
        self.config = config
        self.dev = ori_layer.self_attn_layer_norm.weight.device
        self.act_scales = act_scales
        self.i = i

        self.self_attn = QuantOPTAttention(
            org_module=ori_layer.self_attn,
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            weight_quant = self.weight_quant,
            act_quant = self.act_quant,
            quantize_bmm_input = self.quantize_bmm_input,
            act_scales=self.act_scales,
            i=self.i,
            bias=True
        )
        self.do_layer_norm_before = self.config.do_layer_norm_before
        self.dropout = self.config.dropout
        self.fc1_input_scales = self.act_scales[f"model.decoder.layers.{self.i}.fc1"]["input"]
        self.fc2_input_scales = self.act_scales[f"model.decoder.layers.{self.i}.fc2"]["input"]

        self.self_attn_layer_norm = LayerNorm(ori_layer.self_attn_layer_norm)
        self.fc1 = W8A8Linear_QKVFc1.from_float(ori_layer.fc1, input_scales = self.fc1_input_scales,
                                                weight_quant=self.weight_quant, act_quant=self.act_quant)  # tensor & channel
        self.fc2 = W8A8Linear_OFc2.from_float(ori_layer.fc2, input_scales = self.fc2_input_scales,
                                                weight_quant=self.weight_quant, act_quant='per_token')  # tensor & token
        self.final_layer_norm = LayerNorm(ori_layer.final_layer_norm)
        # self.type = ori_layer.fc1.weight.dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:

        # Self Attention
        hidden_states = hidden_states.to(self.dev)
        attention_mask = attention_mask.to(self.dev)
        residual = hidden_states

        if self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)


        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=0.0, training=False)

        hidden_states = residual + hidden_states
        # print(self.do_layer_norm_before)
        if not self.do_layer_norm_before:
            hidden_states = self.self_attn_layer_norm(hidden_states)

        # Fully Connected
        hidden_states_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
        residual = hidden_states

        # residual.add_(hidden_states.to(residual.dtype))
        if self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)
        # hidden_states = self.final_layer_norm(hidden_states.float()).to(self.type)

        hidden_states = self.fc1(hidden_states)
        hidden_states = F.relu(hidden_states)

        hidden_states = self.fc2(hidden_states)
        # hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = (residual + hidden_states).view(hidden_states_shape)
        # residual.add_(hidden_states.to(residual.dtype))
        if not self.do_layer_norm_before:
            hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs