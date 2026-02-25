# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Run model with xlite graph on NPU."""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Dict, Optional, Union

import numpy as np
import torch

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

try:
    from xlite._C import Runtime, ModelConfig, Model, ModelAttnMeta
    XLITE_AVAILABLE = True
except ImportError:
    XLITE_AVAILABLE = False
    logger.warning("xlite is not available. Install xlite to use xlite graph runner.")


class XliteGraphRunner(CudaGraphRunner):
    """A XliteGraphRunner runs forward pass of a model with xlite graph on NPU."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.model_runner = model_runner
        self.xlite_model = None
        self.xlite_rt = None
        self.xlite_kv_cache = None
        self.xlite_config = None
        
        if not XLITE_AVAILABLE:
            raise ImportError("xlite is not available. Please install xlite first.")
        
        self._init_xlite()

    def _init_xlite(self):
        """Initialize xlite runtime and model."""
        try:
            model_config = self.model_runner.model_config
            
            xlite_config = ModelConfig()
            xlite_config.vocabSize = model_config.vocab_size
            xlite_config.hiddenSize = model_config.hidden_size
            xlite_config.nLayers = model_config.num_hidden_layers
            xlite_config.nHeads = model_config.num_attention_heads
            xlite_config.nKvHeads = model_config.num_key_value_heads
            xlite_config.headDim = model_config.head_dim
            xlite_config.intermediateSize = model_config.intermediate_size
            xlite_config.normEps = model_config.rms_norm_eps
            xlite_config.ropeTheta = model_config.rope_theta
            xlite_config.maxBatch = self.model_runner.server_args.max_num_seqs
            xlite_config.maxSeqLen = model_config.max_model_len
            xlite_config.maxM = xlite_config.maxBatch * xlite_config.maxSeqLen
            xlite_config.blockSize = model_config.block_size
            
            attention_arch = model_config.get_attention_arch()
            if attention_arch == AttentionArch.MLA:
                xlite_config.attnType = 1
            else:
                xlite_config.attnType = 0
            
            if hasattr(model_config, "moe_num_experts") and model_config.moe_num_experts > 1:
                xlite_config.nRoutedExperts = model_config.moe_num_experts
                xlite_config.nActExperts = getattr(model_config, "moe_top_k", 1)
            
            xlite_config.defTpSize = self.model_runner.tp_size
            
            self.xlite_config = xlite_config
            
            self.xlite_rt = Runtime()
            self.xlite_model = Model()
            self.xlite_model.init(xlite_config, self.model_runner.tp_rank)
            
            logger.info(f"XliteGraphRunner initialized with config: vocab_size={xlite_config.vocabSize}, "
                       f"hidden_size={xlite_config.hiddenSize}, n_layers={xlite_config.nLayers}")
            
        except Exception as e:
            logger.error(f"Failed to initialize xlite: {e}")
            raise

    def _create_xlite_attn_meta(self, forward_batch: ForwardBatch) -> ModelAttnMeta:
        """Create xlite attention metadata from forward batch."""
        attn_meta = ModelAttnMeta()
        
        attn_meta.lens = forward_batch.seq_lens.tolist()
        attn_meta.cachedLens = forward_batch.seq_lens.tolist()
        
        is_prefills = []
        for i in range(len(forward_batch.seq_lens)):
            is_prefills.append(forward_batch.forward_mode.is_prefill())
        attn_meta.isPrefills = is_prefills
        
        attn_meta.blockTables = []
        for i in range(len(forward_batch.seq_lens)):
            attn_meta.blockTables.append(forward_batch.block_tables[i].tolist())
        
        return attn_meta

    def replay(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        """Run forward pass with xlite."""
        if not skip_attn_backend_init:
            self.replay_prepare(forward_batch, pp_proxy_tensors)
        
        attn_meta = self._create_xlite_attn_meta(forward_batch)
        
        input_ids = forward_batch.input_ids
        
        output_buffer = self.output_buffers.get(self.bs, None)
        if output_buffer is None:
            output_buffer = torch.zeros(
                (input_ids.shape[0], self.model_runner.model_config.vocab_size),
                dtype=torch.float16,
                device=input_ids.device,
            )
        
        freqs_cis = forward_batch.freqs_cis if hasattr(forward_batch, 'freqs_cis') else None
        
        stream = torch.npu.current_stream().cuda_stream
        
        self.xlite_model.forward_and_get_logits(
            self.xlite_rt,
            input_ids,
            attn_meta,
            self.xlite_kv_cache,
            freqs_cis,
            output_buffer,
            stream,
        )
        
        if isinstance(output_buffer, LogitsProcessorOutput):
            return LogitsProcessorOutput(
                next_token_logits=output_buffer.next_token_logits[: input_ids.shape[0]],
                hidden_states=(
                    output_buffer.hidden_states[: input_ids.shape[0]]
                    if output_buffer.hidden_states is not None
                    else None
                ),
            )
        else:
            return LogitsProcessorOutput(
                next_token_logits=output_buffer[: input_ids.shape[0]],
                hidden_states=None,
            )
