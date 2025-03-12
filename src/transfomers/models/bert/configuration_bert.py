from collections import OrderedDict
from typing import Mapping
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

logger = logging.get_logger(__name__)


class BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`BertModel`] or a [`TFbertModel`]. It is used to
    instantiate a BERT Model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the default will yield a similar configuraton to that of the BERT
    architecture.

    Configuration object inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PreTrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            `input_ids` passed when calling [`BertModel`] or [`TFBertModel`]
        hidden_size (`int`, *optional*, defaults to 768)
            Dimensionlity of the encoder layers and the pooler layer.
        num_hidden_layers (`int`, *optional*, defaults to 12)
            Number of hidden layers in the Transformer encoder
        num_attention_heads (`int`, *optional*, defaults to 12)
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072)
            Dimensionality of the "intermediate"(often named feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`)
            The non-linear activation function (function or string) in the encoder and pooler. If string `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1)
            The dropout probability for all fully connected layers in the embedding, encoder and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1)
            The dropout ratio for the attention probabilities
        max_position_embeddings (`int`, *optional* defaults to 512)
            The maximum sequence length that this model might ever be used with. Typically sert this to something large just in case
        type_vocab_size (`int`, *optional*, default to 2)
            The vocabulary size of the `token_type_ids` passed when calling [`BertModel`] or [`TFBertModel`]
        initializer_range (`float`, *optional*, defaults to 2)
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12)
            The epsilon used by the layer normalization layers.
        position_embedding_type (`str`, *optional* defaults to `"absolute"`)
            Type of position embeddings,. choose one of `"absolute"`, `"relative_key"`. For
            positional embeddings use `"absolute"`.
        is_decoder (`bool`, *optional*, defaults to `False`)
            Wheteer the model is used as a decoder or not. If `False`m, the model is used as encoder
        use_cache (`bool`, *optional*, defaults to `True`)
            Whether or not the model should return the last key/values attentions (not used by all models.) Only if
            relevant if `config.is_decoder=True`
        classifier_dropout (`float`, *optional*):
            The dropout ration for the classification head.
    Examples:
    ```python
    >>> from transformers import BertConfig, BertModel
    >>> configuration = BertConfig()
    >>> model = BertModel(configuration)
    >>> configurartion = model.config
    ```"""

    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classfier_dropout = classifier_dropout


class BertOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
            ]
        )


__all__ = ["BertConfig", "BertOnnxConfig"]
