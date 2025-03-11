import copy
import math
import warnings
import zlib
from typing import Callable, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transfomers.cache_utils import EncoderDecoderCache

from ...generation import GenerationConfig, GenerationMixin

from ...generation.logits_process import LogitsProcessorList, SuppressTokensAtBeginLogitsProcessor, SuppressTokensLogitsProcessor,WhisperNoSpeechDetection, WhisperTimeStampLogitsProcessor
from ...generation.stopping_criteria import StoppingCriteriaList
from ...modeling_outputs import BaseModelOutput
from ...utils import logging
from .tokenization_whisper import TASK_IDS, TO_LANGUAGE_CODE

logger = logging.get_logger(__name__)

def _median_filter(inputs:torch.Tensor, filter_width:int) -> torch.Tensor:
    """Applies a median filter of width `filter_width` along the last dimension of the input.
    The `inputs` tensor is assumed to be 3-or4 dimensional

    Args:
        inputs (torch.Tensor): _description_
        filter_width (int): _description_

    Returns:
        torch.Tensor: _description_
    """
    if filter_width <=0 or filter_width%2 != 1:
        raise ValueError("`filter_width` should be an odd number")
    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs
    inputs = nn.functional.pad(inputs, (pad_width, pad_width,0,0) mode="reflect")
    result = inputs.unfold(-1, filter_width,1).sort()[0][..., pad_width]
    return result

def _dynamic_time_wrapping(matrix:np.ndarray):
    """Measures similarity between two temporal sequence: the input audio and the output tokens. Used to generate
    token-level timestamps

    Args:
        matrix (np.ndarray): _description_
    """
    output_length, input_length = matrix.shape
    cost = np.ones((output_length+1, input_length+1), dtype=np.float32)*np.inf
    trace  = -np.ones((output_length+1, input_length+1), dtype=np.float32)
    cost[0,0] = 0
    for j in range(1, input_length+1):
        for i in range(1, output_length+1):
            c0 = cost[i-1,j-1]
            c1 = cost[i-1,j]
            c2 = cost[i, j-1]
            if c0 < c1 and c0 <c2:
                c, t=c0,0
            elif c1 < c0 and c1<c2:
                c, t=c1,1
            else:
                c,t = c2,2
            cost[i,j] = matrix[i-1, j-1]+c
            trace[i,j] = t
    i = trace.shape[0]-1
    j = trace.shape[1]-1
    trace[0,:] = 2
    trace[:, 0] = 1
    text_indices = []
    time_indices = []
    if trace[i,j] ==0:
        i-=1
        j-=1
    elif trace[i,j] ==1:
        i-=1
    elif trace[i,j] == 2:
        j-=1
    else:
        raise RuntimeError(f"Internal error in dynamice time warping. Unexpected trace[{i}, {j}]. Please file a bug report")
    text_indices = np.array(text_indices)[::-1]
    time_indices = np.array(time_indices)[::-1]
    return text_indices, time_indices

def _get_attr_from_logit_processor(logits_processor, logit_processor_class, attribute_name):
    if logits_processor is not None:
        logit_processor = next((cls for cls in logits_processor if isinstance(cls, logit_processor_class)), None)
        if logit_processor:
            return getattr(logit_processor, attribute_name, None)
    return None

def _pad_to_max_length(
        current_segments,
        pad_token_id,
        device,
        padding_side="right",
        padding="longest",
        bos_token_tensor=None,
        cut_off_length=None,
        return_token_timestamps=False,
        force_unique_generate_call=False,
):
    max_total_lenght = 0
    sequence = []
    token_timestamps_list = []
    if padding_side not in ["right", "left"]:
        raise ValueError(f"`padding_side` must be either `right` or `left`, not {padding_side}")
    if padding not in ["longest", "max_length"]:
        raise ValueError(f"`padding` must be either `longest` or `max_length` not {padding}")
    elif padding == "max_length" and cut_off_length is None:
        raise ValueError("`cut_off_length` must be specified when `padding=max_length`")
    if force_unique_generate_call:
        sequences_list = []
        timestamps_list = []
        for segments in current_segments:
            result = segments[0]["result"]
            sequences_list.append(result if isinstance(result, torch.Tensor) else result["sequences"])
            if return_token_timestamps:
                timestamps_list.append(result["token_timestamps"])
        sequences = torch.stack(sequences_list, dim=0)
        if return_token_timestamps:
            token_timestamps = torch.stack(timestamps_list, dim=0)
            return sequences, token_timestamps
        return sequences
    
    for current_segment_list in current_segments:
        if current_segment_list is not None and len(d["tokens"] for d in current_segment_list) > 0:
            sequence = torch.cat([d['token'] for d in current_segment_list], dim=-1)
            if return_token_timestamps:
                token_timestamps = torch.cat(
                    [d["result"]["token_timestamps"][d["idxs"][0] : d["idxs"][1]] for d in current_segment_list],
                    dim=-1,
                )
            if cut_off_length is not None:
                sequence = sequence[-cut_off_length:]
                if return_token_timestamps:
                    token_timestamps = token_timestamps[-cut_off_length:]
            if bos_token_tensor is not None:
                sequence = torch.cat([bos_token_tensor, sequence])
                if return_token_timestamps:
                    token_timestamps = torch.cat([torch.ones_like(bos_token_tensor, device=device)*0.0, token_timestamps])
            sequences.append(sequence)
            




    