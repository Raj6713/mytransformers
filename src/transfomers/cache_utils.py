import copy
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from packaging import version

from .configuration_utils import PretrainedConfig
from .utils import is_hqq_available, is_optimum_quanto_available, is_torchdynamo_compiling, logging
from .utils.deprecation import deprecate_kwarg

if is_hqq_available():
    from hqq.core.quantize import Quantizer as HQQQuantizer

logger = logging.get_logger(__name__)

class Cache(torch.nn.Module):
    is_compileable = False

    def __init__(self):
        super().__init__()
    def update(self, key_states:torch.Tensor,value_states:torch.Tensor, layer_idx:int, cache_kwargs:Optional[Dict[str,Any]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            key_states (torch.Tensor): _description_
            value_states (torch.Tensor): _description_
            layer_idx (int): _description_
            cache_kwargs (Optional[Dict[str,Any]], optional): _description_. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: _description_
        """
        raise NotImplementedError("Make sure to implement `update` in a subclass.")
    
    def get_seq_length(self, layer_idx:Optional[int]=0) -> int:
        """_summary_

        Args:
            layer_idx (Optional[int], optional): _description_. Defaults to 0.

        Returns:
            int: _description_
        """
        raise NotImplementedError("Make sure to implement `seq_length` in a subclass.")
    
    def get_max_cache_shape(self)-> Optional[int]:
        """_summary_

        Returns:
            Optional[int]: _description_
        """
        raise NotImplementedError("Make sure to implement `max_cache_shape` in a subclass.")
    
    def get_usable_length(self, new_seq_length:int, layer_idx:Optional[int]=0) -> int:
        max_length = self.get_max_cache_shape()
        previous_seq_length = self.get_seq_length(layer_idx)
        if max_length is not None and previous_seq_length +new_seq_length > max_length:
            return max_length -new_seq_length
        return previous_seq_length

    def reorder_cache(self, beam_idx:torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx] != []:
                device = self.key_cache[layer_idx].device
                self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            if self.value_cache[layer_idx] != []:
                device = self.value_cache[layer_idx].device
                self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))
    
    @property
    def seen_tokens(self):
        logger.warning_once(
            "The `seen_token` attribute is deprecated and will be removed in v4.41. Use the `cache_position`"
            "model input instead."
        )
        if hasattr(self, "_seen_tokens"):
            return self._seen_tokens
        else:
            return None
        
@dataclass
class CacheConfig:
    cache_implementation: None
    
    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        """_summary_

        Args:
            config_dict (_type_): _description_
        """
        config = cls(**config_dict) 
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key,value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)
        return config
    
    def to_json_file(self, json_file_path:Union[str, os.PathLike]):
        """_summary_

        Args:
            json_file_path (Union[str, os.PathLike]): _description_
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            config_dict = self.to_dict()
            json_string = json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
            writer.write(json_string)
    
    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.__dict__)
    
    def __iter__(self):
        for attr, value in copy.deepcopy(self.__dict__).items():
            yield attr, value
            