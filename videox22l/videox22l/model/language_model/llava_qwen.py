#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import random
random.seed(42)
from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn

import transformers
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
# from ...constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from videox22l.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM


import inspect
import math
import warnings
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
# from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast

from .modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)

from transformers.modeling_utils import PreTrainedModel

from transformers.integrations import is_deepspeed_zero3_enabled
from .configuration_qwen2 import Qwen2Config
from .modeling_beacon import Memory
from videox22l.train.modeling_utils import optional_grad_ctx, compute_loss, BeaconModelOutput
import pdb
import psutil
from loguru import logger as eval_logger

import time

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

logger = logging.get_logger(__name__)

torch.utils.checkpoint.set_checkpoint_debug_enabled(True)

_CHECKPOINT_FOR_DOC = "Qwen/Qwen2-7B-beta"
_CONFIG_FOR_DOC = "Qwen2Config"

QWEN2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Qwen/Qwen2-7B-beta",
    # See all Qwen2 models at https://huggingface.co/models?filter=qwen2
]

def monitor_cpu_usage():
    # 获取当前 CPU 使用率
    return psutil.cpu_percent(interval=1)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Qwen2
class Qwen2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class Qwen2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, q, k, position_ids):
        seq_len = max(position_ids.max().item() + 1, k.shape[2])

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=k.device, dtype=k.dtype)

        # batch_size, 1, key_len, head_dim
        k_cos = self.cos_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)
        k_sin = self.sin_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)

        q_cos = k_cos[..., -q.shape[2]:, :]
        q_sin = k_sin[..., -q.shape[2]:, :]

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed


class Qwen2LinearScalingRotaryEmbedding(Qwen2RotaryEmbedding):
    """Qwen2RotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class Qwen2DynamicNTKScalingRotaryEmbedding(Qwen2RotaryEmbedding):
    """Qwen2RotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(self, dim, max_position_embeddings=32768, base=10000, device=None, scaling_factor=1.0):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)


class Qwen2YarnRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0, beta_slow=2, beta_fast=128):
        super().__init__()

        self.base = base
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.beta_slow = beta_slow
        self.beta_fast = beta_fast
        self.max_position_embeddings = max_position_embeddings

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=device, dtype=torch.get_default_dtype()
        )

    def _get_factor(self, device, dtype):
        # the dimension whose index is smaller than fast_dim rotates more than beta_fast
        fast_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_fast)) / math.log(self.base))
        fast_dim = max(math.floor(fast_dim), 0)
        # the dimension whose index is bigger than slow_dim rotates less than beta_slow
        slow_dim = self.dim / 2 * (math.log(self.max_position_embeddings / (2 * math.pi * self.beta_slow)) / math.log(self.base))
        slow_dim = min(math.ceil(slow_dim), self.dim - 1)

        if fast_dim == slow_dim:
            slow_dim += 0.001

        # NOTE: very important to use full precision here so that the factor is correct
        dim_arange = torch.arange(0, self.dim // 2, device=device, dtype=torch.float32)
        dim_factor = (dim_arange - fast_dim) / (slow_dim - fast_dim)
        dim_factor = torch.clamp(dim_factor, 0, 1)

        # align with the paper notation
        return (1 - dim_factor)

    def _get_temperature(self):
        if self.scaling_factor <= 1:
            return 1.0
        return 0.07 * math.log(self.scaling_factor) + 1.0
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        dim_arange = torch.arange(0, self.dim, 2, device=device) / self.dim
        # dim / 2
        freq = self.base ** dim_arange
        theta = 1 / freq
        interleave_theta = theta / self.scaling_factor

        factor = self._get_factor(device, dtype)
        yarn_theta = factor * theta + (1 - factor) * interleave_theta
        self.register_buffer("inv_freq", yarn_theta, persistent=False)

        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        # get attention temperature
        temperature = self._get_temperature()

        self.register_buffer("cos_cached", (emb.cos() * temperature).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * temperature).to(dtype), persistent=False)
        self.max_seq_len_cached = seq_len
    
    def forward(self, q, k, position_ids):
        seq_len = max(position_ids.max().item() + 1, k.shape[2])

        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self.scaling_factor = seq_len / self.max_position_embeddings
            self._set_cos_sin_cache(seq_len=seq_len, device=k.device, dtype=k.dtype)

        k_cos = self.cos_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)
        k_sin = self.sin_cached[position_ids].to(dtype=k.dtype).unsqueeze(1)

        q_cos = k_cos[..., -q.shape[2]:, :]
        q_sin = k_sin[..., -q.shape[2]:, :]

        q_embed = (q * q_cos) + (rotate_half(q) * q_sin)
        k_embed = (k * k_cos) + (rotate_half(k) * k_sin)
        return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.Qwen2MLP with Qwen2->Qwen2
class Qwen2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

        if "mlp" in config.beacon_param:            
            self.beacon_up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.beacon_up_proj.weight.data.zero_()
            self.beacon_up_proj._is_hf_initialized = True

            self.beacon_down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            self.beacon_down_proj.weight.data.zero_()
            self.beacon_down_proj._is_hf_initialized = True

    def _init_beacon_proj(self, missing_keys):
        """Initialize the beacon projection weight with that of the ordinal projection."""
        if "mlp" in self.config.beacon_param:
            if is_deepspeed_zero3_enabled():
                # FIXME: after deepspeed initialization, some weights becomes non-zero
                # For Mistral, there are rows that are full of zeros
                # For Mistral, there are values bigger than 1e29...

                import deepspeed
                params = [self.up_proj.weight, self.down_proj.weight, self.beacon_up_proj.weight, self.beacon_down_proj.weight]
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    if (self.beacon_up_proj.weight.sum(-1) == 0).any() or (self.beacon_up_proj.weight > 1e29).any():
                        self.beacon_up_proj.weight.data[:] = self.up_proj.weight.data
                        self.beacon_down_proj.weight.data[:] = self.down_proj.weight.data
            else:
                if any("beacon_up_proj" in missing_key for missing_key in missing_keys):
                    # only copy the value in-place, without tieing the weight
                    self.beacon_up_proj.weight.data[:] = self.up_proj.weight.data
                    self.beacon_down_proj.weight.data[:] = self.down_proj.weight.data

    def forward(self, x, beacon_size, beacon_indices):
        if "mlp" in self.config.beacon_param:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            if beacon_size > 0:
                cur_beacon_indices = beacon_indices[-x.shape[1]:]
                ordinal_hidden_states = x[:, cur_beacon_indices == 0]
                beacon_hidden_states = x[:, cur_beacon_indices == 1]

                ordinal_down_proj = self.down_proj(self.act_fn(self.gate_proj(ordinal_hidden_states)) * self.up_proj(ordinal_hidden_states))
                beacon_down_proj = self.beacon_down_proj(self.act_fn(self.gate_proj(beacon_hidden_states)) * self.beacon_up_proj(beacon_hidden_states))

                down_proj = beacon_down_proj.new_ones(x.shape)
                down_proj[:, beacon_indices == 0] = ordinal_down_proj
                down_proj[:, beacon_indices == 1] = beacon_down_proj
            else:
                down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()

        # NOTE: add extra parameters for beacon tokens
        # skip post initialization to speed up loading
        if "q" in config.beacon_param:
            self.beacon_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.q_proj.bias is not None)
            # NOTE: initialize the beacon parameters as zero
            self.beacon_q_proj.weight.data.zero_()
            self.beacon_q_proj._is_hf_initialized = True
        if "k" in config.beacon_param:
            self.beacon_k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.k_proj.bias is not None)
            self.beacon_k_proj.weight.data.zero_()
            self.beacon_k_proj._is_hf_initialized = True
        if "v" in config.beacon_param:
            self.beacon_v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.v_proj.bias is not None)
            self.beacon_v_proj.weight.data.zero_()
            self.beacon_v_proj._is_hf_initialized = True
        if "o" in config.beacon_param:
            self.beacon_o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.o_proj.bias is not None)
            self.beacon_o_proj.weight.data.zero_()
            self.beacon_o_proj._is_hf_initialized = True

        if "retrieval_q_proj" in config.beacon_param:
            # 检索 q proj
            self.retrieval_q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.q_proj.bias is not None)
            # NOTE: initialize the beacon parameters as zero
            self.retrieval_q_proj.weight.data.zero_()
            self.retrieval_q_proj._is_hf_initialized = True

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = Qwen2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = Qwen2LinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = Qwen2DynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn":
                self.rotary_emb = Qwen2YarnRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn-t":
                self.rotary_emb = Qwen2YarnDynamicTemperatureRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "yarn-t-logn":
                self.rotary_emb = Qwen2YarnDynamicTemperatureLogNRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _init_beacon_proj(self, missing_keys):
        """Initialize the beacon projection weight with that of the ordinal projection."""
        beacon_param = self.config.beacon_param        
        
        if is_deepspeed_zero3_enabled():
            # FIXME: after deepspeed initialization, some weights becomes non-zero
            # For Mistral, there are rows that are full of zeros
            # For Mistral, there are values bigger than 1e29...

            import deepspeed
            if "q" in beacon_param:
                params = [self.beacon_q_proj.weight, self.q_proj.weight]
                if self.q_proj.bias is not None:
                    params.extend([self.beacon_q_proj.bias, self.q_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_q_proj.weight.sum(-1) == 0).any() or (self.beacon_q_proj.weight > 1e29).any():
                        self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                        if self.q_proj.bias is not None:
                            self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in beacon_param:
                params = [self.beacon_k_proj.weight, self.k_proj.weight]
                if self.k_proj.bias is not None:
                    params.extend([self.beacon_k_proj.bias, self.k_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_k_proj.weight.sum(-1) == 0).any() or (self.beacon_k_proj.weight > 1e29).any():
                        self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                        if self.k_proj.bias is not None:
                            self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param:
                params = [self.beacon_v_proj.weight, self.v_proj.weight]
                if self.v_proj.bias is not None:
                    params.extend([self.beacon_v_proj.bias, self.v_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_v_proj.weight.sum(-1) == 0).any() or (self.beacon_v_proj.weight > 1e29).any():
                        self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                        if self.v_proj.bias is not None:
                            self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in beacon_param:
                params = [self.beacon_o_proj.weight, self.o_proj.weight]
                if self.o_proj.bias is not None:
                    params.extend([self.beacon_o_proj.bias, self.o_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.beacon_o_proj.weight.sum(-1) == 0).any() or (self.beacon_o_proj.weight > 1e29).any():
                        self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                        if self.o_proj.bias is not None:
                            self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data

            if "retrieval_q_proj" in beacon_param:
                params = [self.retrieval_q_proj.weight, self.q_proj.weight]
                if self.q_proj.bias is not None:
                    params.extend([self.retrieval_q_proj.bias, self.q_proj.bias])
                with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                    # FIXME: after deepspeed initialization, some weights becomes non-zero, but there are rows that are full of zeros
                    if (self.retrieval_q_proj.weight.sum(-1) == 0).any() or (self.retrieval_q_proj.weight > 1e29).any():
                        self.retrieval_q_proj.weight.data[:] = self.q_proj.weight.data
                        if self.q_proj.bias is not None:
                            self.retrieval_q_proj.bias.data[:] = self.q_proj.bias.data

        else:

            # only copy the value in-place, without tieing the weight
            if "q" in beacon_param and any("beacon_q_proj" in missing_key for missing_key in missing_keys):
                # FIXME: some beacon weights are not initialized as zero for mistral model, why? 
                # if (self.beacon_q_proj.weight == 0).all():
                    self.beacon_q_proj.weight.data[:] = self.q_proj.weight.data
                    if self.q_proj.bias is not None:
                        self.beacon_q_proj.bias.data[:] = self.q_proj.bias.data
            if "k" in beacon_param and any("beacon_k_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_k_proj.weight == 0).all():
                    self.beacon_k_proj.weight.data[:] = self.k_proj.weight.data
                    if self.k_proj.bias is not None:
                        self.beacon_k_proj.bias.data[:] = self.k_proj.bias.data
            if "v" in beacon_param and any("beacon_v_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_v_proj.weight == 0).all():
                    self.beacon_v_proj.weight.data[:] = self.v_proj.weight.data
                    if self.v_proj.bias is not None:
                        self.beacon_v_proj.bias.data[:] = self.v_proj.bias.data
            if "o" in beacon_param and any("beacon_o_proj" in missing_key for missing_key in missing_keys):
                # if (self.beacon_o_proj.weight == 0).all():
                    self.beacon_o_proj.weight.data[:] = self.o_proj.weight.data
                    if self.o_proj.bias is not None:
                        self.beacon_o_proj.bias.data[:] = self.o_proj.bias.data

            # init retrieval_q_proj
            if "retrieval_q_proj" in beacon_param and any("retrieval_q_proj" in missing_key for missing_key in missing_keys):
                # FIXME: some beacon weights are not initialized as zero for mistral model, why? 
                # if (self.beacon_q_proj.weight == 0).all():
                self.retrieval_q_proj.weight.data[:] = self.q_proj.weight.data
                if self.q_proj.bias is not None:
                    self.retrieval_q_proj.bias.data[:] = self.q_proj.bias.data
            

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def qkv_proj_with_beacon(self, hidden_states, beacon_size, beacon_indices):
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-hidden_states.shape[1]:]

            ordinal_hidden_states = hidden_states[:, cur_beacon_indices == 0]
            beacon_hidden_states = hidden_states[:, cur_beacon_indices == 1]

            if "q" in self.config.beacon_param:
                ordinal_query_states = self.q_proj(ordinal_hidden_states)
                beacon_query_states = self.beacon_q_proj(beacon_hidden_states)
                query_states = beacon_query_states.new_zeros((ordinal_query_states.shape[0], cur_beacon_indices.shape[0], ordinal_query_states.shape[2]))
                query_states[:, cur_beacon_indices == 0] = ordinal_query_states
                query_states[:, cur_beacon_indices == 1] = beacon_query_states
                # NOTE: replicate hidden states for beacon tokens in case of parallel windows
                if (cur_beacon_indices == 2).any():
                    query_states[:, cur_beacon_indices == 2] = beacon_query_states[:, :(cur_beacon_indices == 2).sum()]

            else:
                query_states = self.q_proj(hidden_states)

            if "k" in self.config.beacon_param:
                ordinal_key_states = self.k_proj(ordinal_hidden_states)
                beacon_key_states = self.beacon_k_proj(beacon_hidden_states)
                key_states = beacon_key_states.new_zeros((ordinal_key_states.shape[0], cur_beacon_indices.shape[0], ordinal_key_states.shape[2]))
                key_states[:, cur_beacon_indices == 0] = ordinal_key_states
                key_states[:, cur_beacon_indices == 1] = beacon_key_states
                # NOTE: replicate hidden states for beacon tokens in case of parallel windows
                if (cur_beacon_indices == 2).any():
                    key_states[:, cur_beacon_indices == 2] = beacon_key_states[:, :(cur_beacon_indices == 2).sum()]

            else:
                key_states = self.k_proj(hidden_states)
            
            if "v" in self.config.beacon_param:
                ordinal_value_states = self.v_proj(ordinal_hidden_states)
                beacon_value_states = self.beacon_v_proj(beacon_hidden_states)
                value_states = beacon_value_states.new_zeros((ordinal_value_states.shape[0], cur_beacon_indices.shape[0], ordinal_value_states.shape[2]))
                value_states[:, cur_beacon_indices == 0] = ordinal_value_states
                value_states[:, cur_beacon_indices == 1] = beacon_value_states
                # NOTE: replicate hidden states for beacon tokens in case of parallel windows
                if (cur_beacon_indices == 2).any():
                    value_states[:, cur_beacon_indices == 2] = beacon_value_states[:, :(cur_beacon_indices == 2).sum()]
            else:
                value_states = self.v_proj(hidden_states)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        return query_states, key_states, value_states


    def k_proj_with_retrieval(self, hidden_states, beacon_size, beacon_indices):
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-hidden_states.shape[1]:]

            beacon_hidden_states = hidden_states[:, cur_beacon_indices == 1]

            key_states_for_retrieval = self.k_proj_retrieval(beacon_hidden_states)

        else:
            key_states_for_retrieval = self.k_proj_retrieval(hidden_states)

        return key_states_for_retrieval

    def o_proj_with_beacon(self, attn_output, beacon_size, beacon_indices):
        if beacon_size > 0:
            # NOTE: when beacon_pos == "interleave", the beacon_indices points to all beacon tokens in the current window (cached activations + input_ids), so we shall slice out the part corresponding to the input_ids
            cur_beacon_indices = beacon_indices[-attn_output.shape[1]:]

            if "o" in self.config.beacon_param:
                ordinal_attn_output = self.o_proj(attn_output[:, cur_beacon_indices == 0])
                beacon_attn_output = self.beacon_o_proj(attn_output[:, cur_beacon_indices == 1])
                attn_output = beacon_attn_output.new_zeros(attn_output.shape)
                attn_output[:, cur_beacon_indices == 0] = ordinal_attn_output
                attn_output[:, cur_beacon_indices == 1] = beacon_attn_output
                # NOTE: replicate hidden states for beacon tokens in case of parallel windows
                # if (cur_beacon_indices == 2).any():
                #     attn_output[:, cur_beacon_indices == 2] = beacon_attn_output[:, :(cur_beacon_indices == 2).sum()]
            else:
                attn_output = self.o_proj(attn_output)
        else:
            attn_output = self.o_proj(attn_output)
        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, beacon_size, beacon_indices = past_key_value

        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class Qwen2SdpaAttention(Qwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    def print_info(self, record_for_print):
        # ANSI escape code for green text
        GREEN = '\033[92m'
        RESET = '\033[0m'

        eval_logger.info(GREEN + '='*100 + RESET)

        layer_idx = record_for_print['layer_idx']
        all_chunks = record_for_print['all_chunks']
        topk_indices = record_for_print['topk_indices']
        topk_values = record_for_print['topk_values']
        original_kv_seq_length = record_for_print['original_kv_seq_length']
        now_kv_seq_length = record_for_print['now_kv_seq_length']
        selected_chunk_idx = record_for_print['selected_chunk_idx']
        lmk_loss = record_for_print['lmk_loss']

        # Printing log messages in green
        eval_logger.info(f'{GREEN}Layer_Idx: {layer_idx}{RESET}')
        eval_logger.info(f'{GREEN}Total Chunks Num: {len(all_chunks)}{RESET}')
        eval_logger.info(f'{GREEN}Chunks: {all_chunks}{RESET}')
        eval_logger.info(f'{GREEN}topk_values: {topk_values}{RESET}')
        eval_logger.info(f'{GREEN}topk_indices: {topk_indices}{RESET}')
        eval_logger.info(f'{GREEN}selected_chunk_idx: {selected_chunk_idx}{RESET}')
        eval_logger.info(f'{GREEN}lmk_loss: {lmk_loss}{RESET}')
        eval_logger.info(f'{GREEN}original_kv_seq_length: {original_kv_seq_length}{RESET}')
        eval_logger.info(f'{GREEN}now_kv_seq_length: {now_kv_seq_length}{RESET}')
        eval_logger.info(GREEN + '='*100 + RESET)

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def LMKLoss(self, scores, tagging, total_pos_num):
        scores = F.log_softmax(scores, dim=1)
        loss = -torch.sum(scores * tagging) / total_pos_num
        #print(f"loss: {loss}")
        return loss

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        memory=None,
        layer_idx=None,
        selected_chunk_idx=None,
        temperature=0.04
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]
        past_key, past_value, beacon_size, beacon_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)


        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        lmk_loss = None


        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)
        # return attn_output, None, past_key_value, lmk_loss
        return attn_output, None, past_key_value, lmk_loss


class Qwen2FlashAttention2(Qwen2Attention):
    """
    Qwen2 flash attention module. This module inherits from `Qwen2Attention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        memory = None,
        layer_idx = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False

        bsz, q_len, _ = hidden_states.size()
        kv_seq_len = hidden_states.shape[-2]    

        past_key, past_value, beacon_size, beacon_indices = past_key_value
        if past_key is not None:
            past_seq_len = past_key.shape[2]
            kv_seq_len += past_seq_len
        else:
            past_seq_len = 0

        query_states, key_states, value_states = self.qkv_proj_with_beacon(hidden_states, beacon_size, beacon_indices)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # return keys and values before rope
        # NOTE: incrementally return keys and values for efficiency 
        past_key_value = (key_states, value_states, beacon_size, beacon_indices)

        if past_key is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key, key_states], dim=2)
            value_states = torch.cat([past_value, value_states], dim=2)

    
        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        # FlashAttention will automatically handle grouped query attention
        # key_states = repeat_kv(key_states, self.num_key_value_groups)
        # value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (Qwen2RMSNorm handles it correctly)

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        attn_output = self._flash_attention_forward(
            query_states, 
            key_states, 
            value_states, 
            attention_mask, 
            q_len, 
            dropout=dropout_rate
        )


        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj_with_beacon(attn_output, beacon_size, beacon_indices)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in Qwen2FlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            attn_output_unpad = flash_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_in_batch_q,
                max_seqlen_k=max_seqlen_in_batch_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            attn_output = flash_attn_func(
                query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=causal
            )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
        )
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


QWEN2_ATTENTION_CLASSES = {
    "eager": Qwen2Attention,
    "sdpa": Qwen2SdpaAttention,
    "flash_attention_2": Qwen2FlashAttention2,
}


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.use_sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. "
                "Please make sure use `attention_mask` instead.`"
            )
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        # NOTE: get beacon_size in case the mlp is included in beacon_param
        past_key, past_value, beacon_size, beacon_indices = past_key_value

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # NOTE: custom code
        if 'memory' in kwargs:
            memory = kwargs['memory']
        else:
            memory = None

        layer_idx = kwargs['layer_idx']
        if 'selected_chunk_idx' in kwargs:
            selected_chunk_idx = kwargs['selected_chunk_idx']
        else:
            selected_chunk_idx = None

        ###add
        # attention_mask = attention_mask.float()
        # Self Attention
        hidden_states, self_attn_weights, present_key_value, lmk_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            memory = memory,
            layer_idx = layer_idx,
            selected_chunk_idx = selected_chunk_idx
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, beacon_size, beacon_indices)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        outputs += (lmk_loss,)

        return outputs


QWEN2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Qwen2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)


class Qwen2PreTrainedModel(PreTrainedModel):
    config_class = Qwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


QWEN2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare Qwen2 Model outputting raw hidden-states without any specific head on top.",
    QWEN2_START_DOCSTRING,
)
class Qwen2Model(Qwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size #152064
        

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # BEACON: add beacon embedding
        self.beacon_embed_tokens = nn.Embedding(1, config.hidden_size, self.padding_idx)
        self.beacon_embed_tokens._is_hf_initialized = True

        print(f"_attn_implementation: {config._attn_implementation}")
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.image_idx=0

    def _init_beacon_embed(self, missing_keys):
        """Initialize the beacon token embedding with that of the eos token."""
        if is_deepspeed_zero3_enabled():
            import deepspeed
            params = [self.beacon_embed_tokens.weight, self.embed_tokens.weight]
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                # deepspeed will initialize the parameters to zero
                if (self.beacon_embed_tokens.weight == 0).all():
                    if self.config.beacon_embed_init == "bos":
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                    elif self.config.beacon_embed_init == "eos":
                        if isinstance(self.config.eos_token_id, list):
                            eos_token_id = self.config.eos_token_id[0]
                        else:
                            eos_token_id = self.config.eos_token_id
                        self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                    else:
                        raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")
        else:
            if any("beacon_embed_tokens" in missing_key for missing_key in missing_keys):
                if self.config.beacon_embed_init == "bos":
                    print(f'初始化 beacon_embed_tokens 为 bos')

                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[self.config.bos_token_id]
                elif self.config.beacon_embed_init == "eos":
                    print(f'初始化 beacon_embed_tokens 为 eos')
                    if isinstance(self.config.eos_token_id, list):
                        eos_token_id = self.config.eos_token_id[0]
                    else:
                        eos_token_id = self.config.eos_token_id
                    self.beacon_embed_tokens.weight.data[:] = self.embed_tokens.weight.data[eos_token_id]
                else:
                    raise NotImplementedError(f"Make sure beacon_embed_init is either eos or bos, found {self.config.beacon_embed_init}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_all_layers=None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_all_layers=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_features:Optional[torch.Tensor] = None,
        memory = None,
        selected_chunk_idx=None
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # BEACON: always use cache
        use_cache = True

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key, past_value, beacon_size, beacon_indices = past_key_values[0]

        # BEACON: separately embed ordinal tokens and beacon tokens because ordinal tokens do not receive gradients
        if beacon_size > 0:
            
            # 获取当前的 beacon 索引
            cur_beacon_indices = beacon_indices[-input_ids.shape[1]:]
            beacon_input_ids = input_ids[:, cur_beacon_indices > 0]

            # 定义特殊标记
            special_token = self.config.vocab_size - 1

            # 初始化 inputs_embeds
            inputs_embeds = torch.zeros(
                (*input_ids.shape, image_features.shape[-1]),
                device=input_ids.device,
                dtype=image_features.dtype
            )

            # 创建布尔掩码，标识 input_ids 中等于 special_token 的位置
            mask = input_ids == special_token
            count = mask.sum()

            if count > 0:
                # 计算需要插入的 image_features 的索引，考虑循环
                image_indices = (self.image_idx + torch.arange(count, device=input_ids.device)) % image_features.shape[0]
                
                # 获取对应的 image_features
                selected_image_features = image_features[image_indices]
                
                # 将选中的 image_features 赋值给 inputs_embeds 中对应的位置
                inputs_embeds[mask] = selected_image_features
                
                # 更新 self.image_idx，确保其在 image_features 的范围内循环
                self.image_idx = (self.image_idx + count) % image_features.shape[0]

            # 对 beacon_input_ids 进行嵌入
            beacon_input_embeds = self.beacon_embed_tokens(beacon_input_ids - self.config.vocab_size)
            inputs_embeds[:, cur_beacon_indices > 0] = beacon_input_embeds
      
        else:
            inputs_embeds = self.embed_tokens(input_ids)
        

        # embed positions
        hidden_states = inputs_embeds

        # print("------------------------------------")
        # print("inputs_embeds",inputs_embeds.shape)
        # print(f"input_ids:          {input_ids}")
        # print(f"beacon_indices:     {beacon_indices}")
        # print(f"position_ids:       {position_ids}")
        # print(f"attention_mask:\n{attention_mask == 0}")
        # print("------------------------------------")
        # x = input()
        # if x == "s":
        #     return

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        # BEACON: still use tuple to organize cache
        next_decoder_cache = () if use_cache else None

        all_lmk_loss = 0

        for idx, decoder_layer in enumerate(self.layers):
            
            if position_ids_all_layers is not None:
                position_ids = position_ids_all_layers[idx]
            if attention_mask_all_layers is not None:
                attention_mask = attention_mask_all_layers[idx]

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # BEACON: slice out the past_key_value of the corresponding layer
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    memory=memory,
                    layer_idx=idx,
                    selected_chunk_idx=selected_chunk_idx
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    memory = memory,
                    layer_idx = idx,
                    selected_chunk_idx=selected_chunk_idx
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if layer_outputs[-1] is not None:
                all_lmk_loss = all_lmk_loss + layer_outputs[-1]

        if all_lmk_loss == 0:
            all_lmk_loss = None

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            lmk_loss=all_lmk_loss
        )




class LlavaQwenConfig(Qwen2Config):
    model_type = "llava_qwen"


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaQwenConfig

    def __init__(self, config):
        # super(Qwen2ForCausalLM, self).__init__(config)
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        config.rope_scaling = None

        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        self.is_the_query_input_ids = None # used for bi-level decoding
    

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_model(self):
        return self.model
    

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Override the default from_pretrained to extend vocab size according to beacon_size."""
        only_lmk_loss = kwargs.pop('only_lmk_loss', False)
        only_next_token_loss = kwargs.pop('only_next_token_loss', False)
        kwargs.update(output_loading_info=True)
        model, loading_info = super().from_pretrained(*args, **kwargs)

        # NOTE: set memory after from_pretrained because there may be another transformer model inside the Memory object, which may cause weird erros during loading
        config = model.config
        model.memory = Memory(
            model_config=config,
            k_seq_dim=2,
            v_seq_dim=2,
            only_lmk_loss=only_lmk_loss,
            only_next_token_loss=only_next_token_loss
        )
        
        missing_keys = loading_info["missing_keys"]
        # NOTE: the beacon parameters may or may not be loaded from the checkpoint
        # if it is loaded from the checkpoint, we should not re-initilize it
        model.model._init_beacon_embed(missing_keys)
        # initialize weights of possible q,k,v,o,mlp
        for layer in model.model.layers:
            layer.self_attn._init_beacon_proj(missing_keys) # TODO check
            layer.mlp._init_beacon_proj(missing_keys)

        return model

    def _native_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_all_layers=None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_all_layers=None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        shift_labels: Optional[bool] = True,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        image_features: Optional[torch.Tensor] = None,
        selected_chunk_idx = None
    ) -> Union[Tuple, BeaconModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # when we directly call _native_forward, the past_key_values would be None
        if past_key_values is None:
            # NOTE: set beacon size to 0 to avoid using any beacon parameters, see Qwen2Attention.forward
            past_key_values = [(None, None, 0, None) for _ in range(self.config.num_hidden_layers)]

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_mask_all_layers=attention_mask_all_layers,
            position_ids=position_ids,
            position_ids_all_layers=position_ids_all_layers,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            image_features=image_features,
            memory=self.memory,
            selected_chunk_idx=selected_chunk_idx
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        batch_loss = None
        valid_token_num = None
        
        # print("labels",labels)
        if labels is not None:
            loss, batch_loss, valid_token_num = compute_loss(logits, labels, shift=shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return BeaconModelOutput(
            loss=loss,
            batch_loss=batch_loss,
            lmk_loss=outputs.lmk_loss,
            valid_token_num=valid_token_num,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # def memory_loop(self):
        
    def cat_tensor(self, list_of_tensors, dim=-1):
        list_of_tensors = [t for t in list_of_tensors if t is not None]
        if len(list_of_tensors) > 1:
            result = torch.cat(list_of_tensors, dim=dim)
        elif len(list_of_tensors) == 1:
            result = list_of_tensors[0]
        else:
            result = None
        return result

    def cal_similarity_cross_attn(self, query_embeds, visual_embeds, block_size, topk=3):
        # Step 1: 提取 query embedding（文本部分）
        query_embedding = query_embeds
        QL, D = query_embedding.shape

        # Step 2: 提取视觉部分 embedding
        visual_embedding_w_time = visual_embeds
        visual_len = visual_embedding_w_time.size(0)

        # Step 3: 计算视觉 token 与 query 的点积相似度
        sim = torch.matmul(visual_embedding_w_time, query_embedding.transpose(0, 1))  # shape [B, VL, QL]
        sim = sim.mean(dim=-1)  # 平均所有 query token → shape [B, VL]

        # Step 4: 遍历每个 block，提取其中的视觉 token 组并计算相似度
        scores = []
        
        # for idx, single_block in enumerate(blocks_positions):
        for start in range(0, visual_len, block_size):
            sim_this_block = []
            end = min(visual_len, start+block_size)
            for frame_start in range(start, end, 144):
                frame_end = frame_start + 144
                frame_sim = sim[frame_start:frame_end]  # shape [group_len]
                frame_sim = frame_sim.mean()  # 可选：mean/max
                sim_this_block.append(frame_sim.item())

            # 如果没有有效 group（为空），给一个低分
            if not sim_this_block:
                scores.append(-float('inf'))
            else:
                # 方式一：用平均值代表 block 分数
                # block_score = mean(sim_this_block)

                # 方式二：用最大值代表 block 分数（更关注最相关区域）
                block_score = max(sim_this_block)

                scores.append(block_score)

        # Step 5: 获取 top-k 分数对应的 block indices
        scores_tensor = torch.tensor(scores, device=query_embeds.device)
        topk_indices = torch.topk(scores_tensor, k=topk, dim=0, largest=True).indices

        return topk_indices.tolist()



    def _beacon_forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        image_features:Optional[torch.Tensor] = None,
        window_context:Optional[list] = None,
        frames_chunks = None,
        selected_chunk_idx = None,
        video_id=None
    ):

        record_input_ids = input_ids.clone()
        if attention_mask is not None:
            record_attention_mask = attention_mask.clone()
        else:
            record_attention_mask = attention_mask
        if labels is not None:
            record_labels = labels.clone()
        else:
            record_labels = labels
        
        if input_ids.shape[-1] == 1 or selected_chunk_idx is None:
            self.memory.gt_chunk_idx = selected_chunk_idx
            self.memory.prepare(
                input_ids=record_input_ids, 
                attention_mask=record_attention_mask, 
                labels=record_labels,
                skip_first=beacon_skip_first,
                skip_last=beacon_skip_last,
            )

            lmk_loss = 0
            while not self.memory.finish:
                # NOTE: Dynamic chunk mechanism
                if self.memory.count!=0:
                    if self.memory.count <= len(window_context):
                        self.memory.config.beacon_window = window_context[self.memory.count-1]
                        self.memory.config.beacon_stride = window_context[self.memory.count-1]
                    else:
                        self.memory.config.beacon_window = 1440
                        self.memory.config.beacon_stride = 1440
                self.memory.count+=1

                input_ids, attention_mask_all_layers, position_ids_all_layers, past_key_values, labels = self.memory.step()

                outputs = self._native_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attention_mask_all_layers=attention_mask_all_layers,
                    position_ids=position_ids,
                    position_ids_all_layers=position_ids_all_layers,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    labels=labels,
                    # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
                    shift_labels=False,
                    image_features=image_features,
                    selected_chunk_idx=selected_chunk_idx
                )
                # update past_key_values
                self.memory.update_memory(outputs.past_key_values)
                # t6 = time.time()
                if labels is not None:
                    self.memory.update_loss(outputs.batch_loss, outputs.valid_token_num)

                if outputs.lmk_loss is not None:
                    lmk_loss = lmk_loss + outputs.lmk_loss

            if lmk_loss == 0:
                lmk_loss = None
            outputs = self.memory.output(outputs, lmk_loss=lmk_loss)
        else:
            record_beacon_activations = {}
            low_cmpr_ratio = self.memory.low_cmpr_ratio
            high_cmpr_ratio = self.memory.high_cmpr_ratio
            for cmpr_ratio in [low_cmpr_ratio,high_cmpr_ratio]: # NOTE: change high
                # print(f'start offline encoding at {cmpr_ratio}')
                self.memory.reset()
                self.memory.config.beacon_ratio = [cmpr_ratio]
                self.memory.gt_chunk_idx = selected_chunk_idx
                self.memory.prepare(
                    input_ids=record_input_ids, 
                    attention_mask=record_attention_mask, 
                    labels=record_labels,
                    skip_first=beacon_skip_first,
                    skip_last=beacon_skip_last,
                )

                # after the first window, one token at a time
                lmk_loss = 0
                while not self.memory.finish:
                    # NOTE: Dynamic chunk mechanism
                    if self.memory.count!=0:
                        if self.memory.count <= len(window_context):
                            self.memory.config.beacon_window = window_context[self.memory.count-1]
                            self.memory.config.beacon_stride = window_context[self.memory.count-1]
                        else:
                            self.memory.config.beacon_window = 1440
                            self.memory.config.beacon_stride = 1440
                    self.memory.count+=1

                    input_ids, attention_mask_all_layers, position_ids_all_layers, past_key_values, labels = self.memory.step()

                    outputs = self._native_forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        attention_mask_all_layers=attention_mask_all_layers,
                        position_ids=position_ids,
                        position_ids_all_layers=position_ids_all_layers,
                        past_key_values=past_key_values,
                        inputs_embeds=inputs_embeds,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        output_hidden_states=output_hidden_states,
                        return_dict=return_dict,
                        labels=labels,
                        # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
                        shift_labels=False,
                        image_features=image_features,
                        selected_chunk_idx=selected_chunk_idx
                    )
                    # update past_key_values
                    self.memory.update_memory(outputs.past_key_values)

                    if self.memory.start_idx >= self.memory.beacon_skip_last and past_key_values[0][-2] != 0: 
                        record_beacon_activations[cmpr_ratio] = self.memory.offload_activations

                    if cmpr_ratio == high_cmpr_ratio and self.memory.start_idx >= self.memory.beacon_skip_last and past_key_values[0][-2] != 0: 

                        start = time.time()
                        # 根据 gt chunk 将高密度载入低密度
                        reload_activations = []
                        for layer_idx in range(self.memory.config.num_hidden_layers):
                            reload_activations.append( record_beacon_activations[high_cmpr_ratio][layer_idx] )
                            if selected_chunk_idx is not None:
                                # NOTE: qin
                                try:
                                    for chunk_idx in selected_chunk_idx:
                                        reload_activations[-1][chunk_idx] = record_beacon_activations[low_cmpr_ratio][layer_idx][chunk_idx]
                                except:
                                    pdb.set_trace()

                        end = time.time()
                        
                        # print(f'time 1: {end - start}s')
                        start = time.time()
                        for layer_idx in range(self.memory.config.num_hidden_layers):
                            key_states_this_layer = [ tmp[0] for tmp in reload_activations[layer_idx] ]
                            val_states_this_layer = [ tmp[1] for tmp in reload_activations[layer_idx] ]
                            self.memory.beacon_activations[layer_idx] = (self.cat_tensor(key_states_this_layer, dim=2).detach(), self.cat_tensor(val_states_this_layer,dim=2).detach())
                        end = time.time()
                        # print(f'time 2: {end - start}s')
                        # cat 为beacon activations 然后替换进去

                    # t6 = time.time()
                    if labels is not None:
                        self.memory.update_loss(outputs.batch_loss, outputs.valid_token_num)

                    if outputs.lmk_loss is not None:
                        lmk_loss = lmk_loss + outputs.lmk_loss

                if lmk_loss == 0:
                    lmk_loss = None

                outputs = self.memory.output(outputs, lmk_loss=lmk_loss)

        return outputs

    def forward(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_features: Optional[torch.FloatTensor] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        window_context: Optional[list] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        selected_chunk_idx=None,
        video_id=None,    # video_id
        frames_chunks=None,
        enable_bi_level_decoding: Optional[bool] = False,
        **kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

        if enable_bi_level_decoding:
            query_len = input_ids.size(1)
            pask_kv_len = past_key_values[0][0].size(2)
            total_len = pask_kv_len + query_len
            position_ids = torch.arange(total_len, device=input_ids.device,dtype=torch.long).unsqueeze(0)

            expected_dtype = past_key_values[0][0].dtype
            attention_mask = torch.zeros(query_len, total_len, dtype=expected_dtype, device=input_ids.device)
            temp_mask_p1 = torch.ones(query_len, pask_kv_len, dtype=torch.bool, device=input_ids.device)
            temp_mask_p2 = torch.ones(query_len, query_len, dtype=torch.bool, device=input_ids.device).tril(diagonal=0)
            combined_mask = torch.cat((temp_mask_p1, temp_mask_p2), dim=1)
            attention_mask.masked_fill_(combined_mask.logical_not(), float("-inf"))
            attention_mask = attention_mask.to(dtype=expected_dtype).expand(1,1,query_len,total_len)

            output = self._native_forward(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        position_ids=position_ids,
                                        past_key_values=past_key_values,
                                        inputs_embeds=inputs_embeds,
                                        labels=labels,
                                        use_cache=use_cache,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
            
            # update kv cache
            output.past_key_values = [ (self.cat_tensor([old_kv[0], new_kv[0]], dim=2), self.cat_tensor([old_kv[1], new_kv[1]], dim=2), 0, None) for old_kv, new_kv in zip(past_key_values, output.past_key_values)]

            self.is_the_query_input_ids = False
            return output
            
        if image_features is None:
            if input_ids.shape[1] != 1:
                image_features, window_context,frames_chunks=self.get_image_features(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes)
                image_features=image_features[0]
        
        num_tokens=image_features.shape[0]
   
        if -200 in input_ids:
            start_value = -200
            if num_tokens !=0:
                insert_index = (input_ids == start_value).nonzero(as_tuple=True)[1][0].item()
                negative_tokens = torch.arange(start_value, start_value - num_tokens, -1, device=input_ids.device)
                if labels !=None:
                    ignore_labels = torch.full((1, num_tokens), -100, device=labels.device, dtype=labels.dtype)
                    before_labels = labels[:, :insert_index]
                    after_labels = labels[:, insert_index + 1:]
                    labels = torch.cat((before_labels, ignore_labels, after_labels), dim=1)

                before_input_ids = input_ids[:, :insert_index]
                after_input_ids = input_ids[:, insert_index + 1:]
                input_ids = torch.cat((before_input_ids, negative_tokens.unsqueeze(0), after_input_ids), dim=1)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            input_ids[input_ids < 0] = self.config.vocab_size-1

        if beacon_skip_first is None:
            beacon_skip_first=14
            beacon_skip_last=beacon_skip_first + num_tokens

        with optional_grad_ctx(with_grad=self.training):
            # we can disable beacon to use the original mistral
            if hasattr(self, "_enable_beacon") and self._enable_beacon == False:
         
                return self._native_forward(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids,
                                            past_key_values=past_key_values,
                                            inputs_embeds=inputs_embeds,
                                            labels=labels,
                                            use_cache=use_cache,
                                            output_attentions=output_attentions,
                                            output_hidden_states=output_hidden_states,
                                            return_dict=return_dict)
            else:
                # print("################")
                return self._beacon_forward(input_ids,
                                            attention_mask,
                                            position_ids,
                                            past_key_values,
                                            inputs_embeds,
                                            labels,
                                            use_cache,
                                            output_attentions,
                                            output_hidden_states,
                                            return_dict,
                                            beacon_skip_first,
                                            beacon_skip_last,
                                            image_features,
                                            window_context,
                                            frames_chunks,
                                            selected_chunk_idx,
                                            video_id)



    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        modalities: Optional[List[str]] = ["image"],
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        selected_chunk_idx: Optional[int] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            image_features, window_context, frames_chunks = self.get_image_features(inputs, position_ids, attention_mask, None, None, images, modalities, image_sizes)
            image_features=torch.stack(image_features).squeeze(0)
            kwargs["image_features"] = image_features
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        # return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)
        # print("generate_id",inputs,image_features.shape)
      
        kwargs["window_context"] = window_context
        kwargs["selected_chunk_idx"] = selected_chunk_idx
        kwargs["frames_chunks"] = frames_chunks

        num_tokens=image_features.shape[0]
      

        if beacon_skip_first is None or beacon_skip_last is None:
            beacon_skip_first = (inputs == -200).nonzero(as_tuple=True)[1].item()
            beacon_skip_last = beacon_skip_first  + num_tokens

        if -200 in inputs:
            start_value = -200
            input_ids=inputs
            if num_tokens !=0:
                insert_index = (input_ids == start_value).nonzero(as_tuple=True)[1][0].item()
                negative_tokens = torch.arange(start_value, start_value - num_tokens, -1, device=input_ids.device)
                before_input_ids = input_ids[:, :insert_index]
                after_input_ids = input_ids[:, insert_index + 1:]
                input_ids = torch.cat((before_input_ids, negative_tokens.unsqueeze(0), after_input_ids), dim=1)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                input_ids[input_ids < 0] = self.config.vocab_size-1
                inputs=input_ids
                # print("new_input_id",inputs)

        return super().generate(position_ids=position_ids, attention_mask=attention_mask,inputs=inputs,beacon_skip_first=beacon_skip_first, beacon_skip_last=beacon_skip_last, **kwargs)


    def prefill_kvs_forward(self, 
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        image_features:Optional[torch.Tensor] = None,
        window_context:Optional[list] = None,
        frames_chunks = None,
        selected_chunk_idx = None,
        video_id=None
    ):

        record_input_ids = input_ids.clone()
        if attention_mask is not None:
            record_attention_mask = attention_mask.clone()
        else:
            record_attention_mask = attention_mask
        if labels is not None:
            record_labels = labels.clone()
        else:
            record_labels = labels
        
        record_beacon_activations = {}
        low_cmpr_ratio = self.memory.low_cmpr_ratio
        high_cmpr_ratio = self.memory.high_cmpr_ratio
        for cmpr_ratio in [low_cmpr_ratio,high_cmpr_ratio]: # NOTE: change high
            # print(f'start offline encoding at {cmpr_ratio}')
            self.memory.reset()
            self.memory.config.beacon_ratio = [cmpr_ratio]
            self.memory.gt_chunk_idx = selected_chunk_idx
            self.memory.prepare(
                input_ids=record_input_ids, 
                attention_mask=record_attention_mask, 
                labels=record_labels,
                skip_first=beacon_skip_first,
                skip_last=beacon_skip_last,
            )

            # after the first window, one token at a time
            lmk_loss = 0
            while not self.memory.finish:
                # NOTE: Dynamic chunk mechanism
                if self.memory.count!=0:
                    if self.memory.count <= len(window_context):
                        self.memory.config.beacon_window = window_context[self.memory.count-1]
                        self.memory.config.beacon_stride = window_context[self.memory.count-1]
                    else:
                        self.memory.config.beacon_window = 1440
                        self.memory.config.beacon_stride = 1440
                self.memory.count+=1

                input_ids, attention_mask_all_layers, position_ids_all_layers, past_key_values, labels = self.memory.step()

                outputs = self._native_forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    attention_mask_all_layers=attention_mask_all_layers,
                    position_ids=position_ids,
                    position_ids_all_layers=position_ids_all_layers,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    labels=labels,
                    # NOTE: the labels have been shifted so that all tokens in the window have the proper loss
                    shift_labels=False,
                    image_features=image_features,
                    selected_chunk_idx=selected_chunk_idx
                )
                # update past_key_values
                self.memory.update_memory(outputs.past_key_values)

                if self.memory.start_idx >= self.memory.beacon_skip_last and past_key_values[0][-2] != 0: 
                    record_beacon_activations[cmpr_ratio] = self.memory.offload_activations

        return self.memory.sink_activations, record_beacon_activations


    @torch.no_grad()
    def prefill_bi_kvs(self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        image_features: Optional[torch.FloatTensor] = None,
        beacon_skip_first: Optional[int] = None,
        beacon_skip_last: Optional[int] = None,
        window_context: Optional[list] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        dpo_forward: Optional[bool] = False,
        cache_position=None,
        selected_chunk_idx=None,
        video_id=None,    # video_id
        frames_chunks=None,
        chunk_size=10,
        **kwargs,
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if image_features is None:
            if input_ids.shape[1] != 1:
                image_features, window_context,frames_chunks = self.get_image_features(input_ids, position_ids, attention_mask, past_key_values, labels, images, modalities, image_sizes,chunk_size=chunk_size)
                image_features=image_features[0]
        
        num_tokens=image_features.shape[0]

        if -200 in input_ids:
            start_value = -200
            if num_tokens !=0:
                insert_index = (input_ids == start_value).nonzero(as_tuple=True)[1][0].item()
                negative_tokens = torch.arange(start_value, start_value - num_tokens, -1, device=input_ids.device)
                if labels !=None:
                    ignore_labels = torch.full((1, num_tokens), -100, device=labels.device, dtype=labels.dtype)
                    before_labels = labels[:, :insert_index]
                    after_labels = labels[:, insert_index + 1:]
                    labels = torch.cat((before_labels, ignore_labels, after_labels), dim=1)

                before_input_ids = input_ids[:, :insert_index]
                after_input_ids = input_ids[:, insert_index + 1:]
                input_ids = torch.cat((before_input_ids, negative_tokens.unsqueeze(0), after_input_ids), dim=1)
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            input_ids[input_ids < 0] = self.config.vocab_size-1

        if beacon_skip_first is None:
            beacon_skip_first=14
            beacon_skip_last=beacon_skip_first + num_tokens

        with optional_grad_ctx(with_grad=self.training):
            return self.prefill_kvs_forward(input_ids,
                                        attention_mask,
                                        position_ids,
                                        past_key_values,
                                        inputs_embeds,
                                        labels,
                                        use_cache,
                                        output_attentions,
                                        output_hidden_states,
                                        return_dict,
                                        beacon_skip_first,
                                        beacon_skip_last,
                                        image_features,
                                        window_context,
                                        frames_chunks,
                                        selected_chunk_idx,
                                        video_id)

    @torch.no_grad()
    def bi_level_decoding(
        self,
        inputs: Optional[torch.Tensor] = None,
        system_kvs_offloaded=None,
        video_bi_kvs_offloaded=None,
        high_cmpr_ratio=32,
        low_cmpr_ratio=2,
        selected_chunk_idx: Optional[List[int]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        mixed_kvs = []
        chunk_num = len(video_bi_kvs_offloaded[high_cmpr_ratio][0])
        for layer_idx in range(0,28):
            system_k = system_kvs_offloaded[layer_idx][0]
            system_v = system_kvs_offloaded[layer_idx][1]

            mixed_k = system_k.to(inputs.device)
            mixed_v = system_v.to(inputs.device)
            for chunk_idx in range(chunk_num):
                if chunk_idx in selected_chunk_idx:
                    reloaded_k = video_bi_kvs_offloaded[low_cmpr_ratio][layer_idx][chunk_idx][0].to(inputs.device)
                    reloaded_v = video_bi_kvs_offloaded[low_cmpr_ratio][layer_idx][chunk_idx][1].to(inputs.device)
                else:
                    reloaded_k = video_bi_kvs_offloaded[high_cmpr_ratio][layer_idx][chunk_idx][0].to(inputs.device)
                    reloaded_v = video_bi_kvs_offloaded[high_cmpr_ratio][layer_idx][chunk_idx][1].to(inputs.device)

                mixed_k = torch.cat([mixed_k, reloaded_k], dim=2)
                mixed_v = torch.cat([mixed_v, reloaded_v], dim=2)

            mixed_kvs.append((mixed_k, mixed_v, 0, None)) # The last two elements are for compatibility with subsequent code and have no actual meaning

        # total_len = mixed_kvs[0][0].size(2) + inputs.size(1)
        # position_ids = torch.arange(total_len, device=inputs.device, dtype=torch.long).unsqueeze(0)
        position_ids = None
        attention_mask = None

        return super().generate(position_ids=position_ids, attention_mask=attention_mask, inputs=inputs, past_key_values=mixed_kvs, enable_bi_level_decoding=True, **kwargs)


    def prepare_inputs_for_generation(
    self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, beacon_skip_first=None, beacon_skip_last=None, enable_bi_level_decoding=False, **kwargs):
        if past_key_values:
            if enable_bi_level_decoding:
                if not self.is_the_query_input_ids:
                    input_ids = input_ids[:, -1:]
            else:
                input_ids = input_ids[:, -1:]
        # print("prepare_ids",input_ids)
        model_inputs = {"input_ids": input_ids}
        model_inputs["beacon_skip_first"]= beacon_skip_first
        model_inputs["beacon_skip_last"]= beacon_skip_last
        model_inputs["past_key_values"] = past_key_values
        model_inputs["enable_bi_level_decoding"] = enable_bi_level_decoding
        model_inputs["position_ids"] = kwargs.get('position_ids', None)
        model_inputs["attention_mask"] = kwargs.get('attention_mask', None)

        if 'image_features' in kwargs:
            model_inputs["image_features"] = kwargs['image_features']
        
        if "window_context" in kwargs:
            model_inputs["window_context"] = kwargs['window_context']

        if "selected_chunk_idx" in kwargs:
            model_inputs["selected_chunk_idx"] = kwargs['selected_chunk_idx']

        if "frames_chunks" in kwargs:
            model_inputs["frames_chunks"] = kwargs['frames_chunks']
        
        # if "frames_chunks" in kwargs:
        #     model_inputs["frames_chunks"] = kwargs['frames_chunks']
        return model_inputs
    
    

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past



AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
