import os
import math
import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...generation import GeneationMixin
from ...modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from ...modeling_outputs import (
    BaseModelOutputWIthPastAndCrossAttentions,
    CausalLMOutputWithCrossAttention,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel, SequenceSummary
from ...pytorch_utils import (
    Conv1D,
    find_pruneable_heads_and_indicies,
    prune_conv1d_layer,
)
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ...utils.model_parallel_utils import assert_device_map, get_device_map
from .configuration_gpt2 import GPT2Config

logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "openai-community/gpt2"
_CONFIG_FOR_DOC = "GPT2Config"


def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """_summary_

    Args:
        model (_type_): _description_
        config (_type_): _description_
        gpt2_checkpoint_path (_type_): _description_
    """
    try:
        import re
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a Tensorflow model in Pytorch, rquires Tensorflow to be installed. Please see"
            "check tensorflow guides online"
        )
        raise
    tf_path = os.path.abspath(gpt2_checkpoint_path)
    logger.info(f"Converting Tensorflow checkpoint from {tf_path}")
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info(f"Loading TF weight {name} with shape {shape}")
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array.squeeze())
    for name, array in zip(name, arrays):
        name = name[6:]
        name = name.split("/")
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+\d+", m_name):
                scope_names = re.split(r"(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] == "w" or scope_names[0] == "g":
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "b":
                pointer = getattr(pointer, "bias")
            elif scope_names[0] == "wpe" or scope_names[0] == "wte":
                pointer = getattr(pointer, scope_names[0])
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, scope_names[0])
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        try:
            if pointer.shape != array.shape:
                raise ValueError(
                    f"Pointer shape {pointer.shape} and array shape {array.shape} mismatched"
                )
        except ValueError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info(f"Initialize Pytorch Weight: {name}")
        pointer.data = torch.from_numpy(array)
    return model


def eager_attention_forward(
    module, query, key, value, attention_mask, head_mask=None, **kwargs
):
    attn_weights = torch.matmul(query, key.transpose(-1, -2))
    if module.scale_attn_weights:
        attn_weights = attn_weights / torch.full(
            [],
            value.size(-1) ** 0.5,
            dtype=attn_weights.dtype,
            device=attn_weights.device,
        )
    if module.scale_attn_by_inverse_layer_idx:
        attn_weights = attn_weights / float(module.layer_idx + 1)
    if not module.is_cross_attention:
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = module.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full(
            [], mask_value, dtype=attn_weights.dtype, device=attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )

    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(value.dtype)
    attn_weights = module.attn_dropout(attn_weights)

    if head_mask is not None:
        attn_weights = attn_weights * head_mask
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2)

    return attn_output, attn_weights

class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(1, 1, max_positions, max_positions),
            persistent=False
        )
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.embed_dim // self.num_heads
        self.split_size = config.embed_dim
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f"{self.num_heads})."
            )
        self.scale_attn_weights = config.scale_attn_weights
        self.is_cross_attention = is_cross_attention
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        if self.is_cross_attention:
            self.c_attn = Conv1D(2*self.embed_dim, self.embed_dim)
            self.q_attn = Conv1D(self.embed_dim, self.embed_dim)
        else:
            self.c_attn = Conv1D(3*self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads)==0:
            return
        heads, index = find_pruneable_heads_and_indicies(heads, self.num_heads, self.head_dim, self.pruned_heads)
        index_attn = torch.cat([index, index+self.split_size, index+(2*self.split_size)])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index,dim=0)
        self.split_size = (self.split_size//self.num_heads) *(self.num_heads-len(heads))
        self.num_heads = self.num_heads - len(heads)
        self.pruned_heads = self.prune_heads.union(heads)

    def _upcast_and_reordered_attn(self, query, key, value, attention_mask=None, head_mask=None):
        bsz, num_heads,q_seq_len, dk = query.size()
        _, _, k_seq_len, _ = key.size()
        attn_weights = torch.empty(bsz*num_heads, q_seq_len, k_seq_len, dtype=torch.float32, device=query.device)
        scale_factor = 1.0
        if self.scale_attn_weights:
            scale_factor/=float(value.size(-1))**0.5
        if self.scale_attn_by_inverse_layer_idx:
            scale_factor /= float(self.layer_idx+1)
        with torch.amp.autocast(query.device.type, enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1,-2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.baddbmm(attn_weights, q.float(), k.float(), beta=0, alpha=scale_factor)
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
        
        if not self.is_cross_attention:
            query_length, key_length = query.size(-2), key.size(-2)
            causal_mask = self.bias[:, :, key_length-query_length:key_length, :key_length]
            mask_value = torch.finfo(attn_weights.dtype).min
            mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
            attn_weights = torch.where(causal_mask, attn_weights, mask_value)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if attn_weights.dtype != torch.float32:
            raise RuntimeError("Error with upcasting, attn_weights does not have dtype torch.float32")
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1,2)
        return attn_output, attn_weights
    
    def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.FloatTensor]]=None,
            attention_mask: Optional[torch.FloatTensor]=None,
            head_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states:Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor]=None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool]=False,
            **kwargs
    ):
        if encoder_hidden_states is not None:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `a_attn` have to be defined"
                    "Pleae make sure to instantaniate class with `GPT2Attention(..., is_cross_attention=True)`"
                )
            query_states = self.q_attn(hidden_states)
            key_states, value_states = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        shape_q = (*query_states.shape[:-1], self.head_dim)
        shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1,2)
        key_states = key_states.view(shape_kv).transpose(1,2)
        value_states = value_states.view(shape_kv).transpose(1,2)
        if layer_past is not None:
            past_key, past_value = layer_past
            key_states = torch.cat((past_key,key_states), dim=-2)
            value_states = torch.cat((past_value, value_states), dim=-2)
        
        if use_cache is True:
            present = (key_states, value_states)
        else:
            present = None
        is_cross_attention = encoder_hidden_states is not None
        is_causal = attention_mask is None and query_states.shape[-2] > 1 and not is_cross_attention

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and (output_attentions or head_mask is not None):
                using_eager = True
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`, Falling back to"
                    "eager attention. This warning can be removed using the argumnet `attn_implemenation=eager` when loading the model")
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implentation]
        
        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask, head_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                head_mask=head_mask,
                dropout = self.attn_dropout.p if self.training else 0.0,
                is_causal=is_causal,
                **kwargs
            )
        attn_output = attn_output.reshape(*attn_output.shape[:-2],-1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)
        return outputs
    
class GPT2MLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states:Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    

class GPT2Block(nn.Module):
    def __init__(self, config, layer_idx=None):
        super().__init__()
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4*hidden_size
        self.ln_l = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(config=config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size,eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(inner_dim, config)

    def forward(self,
                hidden_states: Optional[Tuple[torch.FloatTensor]],
                layer_past: Optional[Tuple[torch.FloatTensor]],
                attention_mask: Optional[torch.FloatTensor]=None,
                head_mask : Optional[torch.FloatTensor]=None,
                encoder_hidden_states:Optional[torch.Tensor]=None,
                encoder_attention_mask:Optional[torch.FloatTensor]=None,
                use_cache:Optional[bool]=False,
                output_attentions:Optional[bool]=False,
                ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor,...]]]]:
        residual = hidden_states
        hidden_states = self.ln_l(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output+residual
        if encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with"
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions
            )
            attn_output = cross_attn_outputs[0]
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,)+outputs
        else:
            outputs = (hidden_states,)+outputs[1:]
        return outputs
