# Copyright 2022 The HuggingFace Team. All rights reserved.
#
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

from typing import TYPE_CHECKING, Optional
import torch
from ..utils import (
    OptionalDependencyNotAvailable,
    _LazyModule,
    is_flax_available,
    is_tf_available,
    is_torch_available,
)


_import_structure = {
    "configuration_utils": [
        "BaseWatermarkingConfig",
        "CompileConfig",
        "GenerationConfig",
        "GenerationMode",
        "SynthIDTextWatermarkingConfig",
        "WatermarkingConfig",
    ],
    "streamers": ["AsyncTextIteratorStreamer", "TextIteratorStreamer", "TextStreamer"],
}

try:
    if not is_torch_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["beam_constraints"] = [
        "Constraint",
        "ConstraintListState",
        "DisjunctiveConstraint",
        "PhrasalConstraint",
    ]
    _import_structure["beam_search"] = [
        "BeamHypotheses",
        "BeamScorer",
        "BeamSearchScorer",
        "ConstrainedBeamSearchScorer",
    ]
    _import_structure["candidate_generator"] = [
        "AssistedCandidateGenerator",
        "CandidateGenerator",
        "EarlyExitCandidateGenerator",
        "PromptLookupCandidateGenerator",
    ]
    _import_structure["logits_process"] = [
        "AlternatingCodebooksLogitsProcessor",
        "ClassifierFreeGuidanceLogitsProcessor",
        "EncoderNoRepeatNGramLogitsProcessor",
        "EncoderRepetitionPenaltyLogitsProcessor",
        "EpsilonLogitsWarper",
        "EtaLogitsWarper",
        "ExponentialDecayLengthPenalty",
        "ForcedBOSTokenLogitsProcessor",
        "ForcedEOSTokenLogitsProcessor",
        "HammingDiversityLogitsProcessor",
        "InfNanRemoveLogitsProcessor",
        "LogitNormalization",
        "LogitsProcessor",
        "LogitsProcessorList",
        "MinLengthLogitsProcessor",
        "MinNewTokensLengthLogitsProcessor",
        "MinPLogitsWarper",
        "NoBadWordsLogitsProcessor",
        "NoRepeatNGramLogitsProcessor",
        "PrefixConstrainedLogitsProcessor",
        "RepetitionPenaltyLogitsProcessor",
        "SequenceBiasLogitsProcessor",
        "SuppressTokensLogitsProcessor",
        "SuppressTokensAtBeginLogitsProcessor",
        "SynthIDTextWatermarkLogitsProcessor",
        "TemperatureLogitsWarper",
        "TopKLogitsWarper",
        "TopPLogitsWarper",
        "TypicalLogitsWarper",
        "UnbatchedClassifierFreeGuidanceLogitsProcessor",
        "WhisperTimeStampLogitsProcessor",
        "WatermarkLogitsProcessor",
    ]
    _import_structure["stopping_criteria"] = [
        "MaxLengthCriteria",
        "MaxTimeCriteria",
        "ConfidenceCriteria",
        "EosTokenCriteria",
        "StoppingCriteria",
        "StoppingCriteriaList",
        "validate_stopping_criteria",
        "StopStringCriteria",
    ]
    _import_structure["utils"] = [
        "GenerationMixin",
        "GreedySearchEncoderDecoderOutput",
        "GreedySearchDecoderOnlyOutput",
        "SampleEncoderDecoderOutput",
        "SampleDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput",
        "BeamSearchDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput",
        "BeamSampleDecoderOnlyOutput",
        "ContrastiveSearchEncoderDecoderOutput",
        "ContrastiveSearchDecoderOnlyOutput",
        "GenerateBeamDecoderOnlyOutput",
        "GenerateBeamEncoderDecoderOutput",
        "GenerateDecoderOnlyOutput",
        "GenerateEncoderDecoderOutput",
    ]
    _import_structure["watermarking"] = [
        "WatermarkDetector",
        "WatermarkDetectorOutput",
        "BayesianDetectorModel",
        "BayesianDetectorConfig",
        "SynthIDTextWatermarkDetector",
    ]

try:
    if not is_tf_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["tf_logits_process"] = [
        "TFForcedBOSTokenLogitsProcessor",
        "TFForcedEOSTokenLogitsProcessor",
        "TFForceTokensLogitsProcessor",
        "TFLogitsProcessor",
        "TFLogitsProcessorList",
        "TFLogitsWarper",
        "TFMinLengthLogitsProcessor",
        "TFNoBadWordsLogitsProcessor",
        "TFNoRepeatNGramLogitsProcessor",
        "TFRepetitionPenaltyLogitsProcessor",
        "TFSuppressTokensAtBeginLogitsProcessor",
        "TFSuppressTokensLogitsProcessor",
        "TFTemperatureLogitsWarper",
        "TFTopKLogitsWarper",
        "TFTopPLogitsWarper",
    ]
    _import_structure["tf_utils"] = [
        "TFGenerationMixin",
        "TFGreedySearchDecoderOnlyOutput",
        "TFGreedySearchEncoderDecoderOutput",
        "TFSampleEncoderDecoderOutput",
        "TFSampleDecoderOnlyOutput",
        "TFBeamSearchEncoderDecoderOutput",
        "TFBeamSearchDecoderOnlyOutput",
        "TFBeamSampleEncoderDecoderOutput",
        "TFBeamSampleDecoderOnlyOutput",
        "TFContrastiveSearchEncoderDecoderOutput",
        "TFContrastiveSearchDecoderOnlyOutput",
    ]

try:
    if not is_flax_available():
        raise OptionalDependencyNotAvailable()
except OptionalDependencyNotAvailable:
    pass
else:
    _import_structure["flax_logits_process"] = [
        "FlaxForcedBOSTokenLogitsProcessor",
        "FlaxForcedEOSTokenLogitsProcessor",
        "FlaxForceTokensLogitsProcessor",
        "FlaxLogitsProcessor",
        "FlaxLogitsProcessorList",
        "FlaxLogitsWarper",
        "FlaxMinLengthLogitsProcessor",
        "FlaxSuppressTokensAtBeginLogitsProcessor",
        "FlaxSuppressTokensLogitsProcessor",
        "FlaxTemperatureLogitsWarper",
        "FlaxTopKLogitsWarper",
        "FlaxTopPLogitsWarper",
        "FlaxWhisperTimeStampLogitsProcessor",
        "FlaxNoRepeatNGramLogitsProcessor",
    ]
    _import_structure["flax_utils"] = [
        "FlaxGenerationMixin",
        "FlaxGreedySearchOutput",
        "FlaxSampleOutput",
        "FlaxBeamSearchOutput",
    ]

if TYPE_CHECKING:
    from .configuration_utils import (
        BaseWatermarkingConfig,
        CompileConfig,
        GenerationConfig,
        GenerationMode,
        SynthIDTextWatermarkingConfig,
        WatermarkingConfig,
    )
    from .streamers import AsyncTextIteratorStreamer, TextIteratorStreamer, TextStreamer

    try:
        if not is_torch_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .beam_constraints import (
            Constraint,
            ConstraintListState,
            DisjunctiveConstraint,
            PhrasalConstraint,
        )
        from .beam_search import (
            BeamHypotheses,
            BeamScorer,
            BeamSearchScorer,
            ConstrainedBeamSearchScorer,
        )
        from .candidate_generator import (
            AssistedCandidateGenerator,
            CandidateGenerator,
            EarlyExitCandidateGenerator,
            PromptLookupCandidateGenerator,
        )
        from .logits_process import (
            AlternatingCodebooksLogitsProcessor,
            ClassifierFreeGuidanceLogitsProcessor,
            EncoderNoRepeatNGramLogitsProcessor,
            EncoderRepetitionPenaltyLogitsProcessor,
            EpsilonLogitsWarper,
            EtaLogitsWarper,
            ExponentialDecayLengthPenalty,
            ForcedBOSTokenLogitsProcessor,
            ForcedEOSTokenLogitsProcessor,
            HammingDiversityLogitsProcessor,
            InfNanRemoveLogitsProcessor,
            LogitNormalization,
            LogitsProcessor,
            LogitsProcessorList,
            MinLengthLogitsProcessor,
            MinNewTokensLengthLogitsProcessor,
            MinPLogitsWarper,
            NoBadWordsLogitsProcessor,
            NoRepeatNGramLogitsProcessor,
            PrefixConstrainedLogitsProcessor,
            RepetitionPenaltyLogitsProcessor,
            SequenceBiasLogitsProcessor,
            SuppressTokensAtBeginLogitsProcessor,
            SuppressTokensLogitsProcessor,
            SynthIDTextWatermarkLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
            TypicalLogitsWarper,
            UnbatchedClassifierFreeGuidanceLogitsProcessor,
            WatermarkLogitsProcessor,
            WhisperTimeStampLogitsProcessor,
        )
        from .stopping_criteria import (
            ConfidenceCriteria,
            EosTokenCriteria,
            MaxLengthCriteria,
            MaxTimeCriteria,
            StoppingCriteria,
            StoppingCriteriaList,
            StopStringCriteria,
            validate_stopping_criteria,
        )
        from .utils import (
            BeamSampleDecoderOnlyOutput,
            BeamSampleEncoderDecoderOutput,
            BeamSearchDecoderOnlyOutput,
            BeamSearchEncoderDecoderOutput,
            ContrastiveSearchDecoderOnlyOutput,
            ContrastiveSearchEncoderDecoderOutput,
            GenerateBeamDecoderOnlyOutput,
            GenerateBeamEncoderDecoderOutput,
            GenerateDecoderOnlyOutput,
            GenerateEncoderDecoderOutput,
            GenerationMixin,
            GreedySearchDecoderOnlyOutput,
            GreedySearchEncoderDecoderOutput,
            SampleDecoderOnlyOutput,
            SampleEncoderDecoderOutput,
        )
        from .watermarking import (
            BayesianDetectorConfig,
            BayesianDetectorModel,
            SynthIDTextWatermarkDetector,
            WatermarkDetector,
            WatermarkDetectorOutput,
        )

    try:
        if not is_tf_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .tf_logits_process import (
            TFForcedBOSTokenLogitsProcessor,
            TFForcedEOSTokenLogitsProcessor,
            TFForceTokensLogitsProcessor,
            TFLogitsProcessor,
            TFLogitsProcessorList,
            TFLogitsWarper,
            TFMinLengthLogitsProcessor,
            TFNoBadWordsLogitsProcessor,
            TFNoRepeatNGramLogitsProcessor,
            TFRepetitionPenaltyLogitsProcessor,
            TFSuppressTokensAtBeginLogitsProcessor,
            TFSuppressTokensLogitsProcessor,
            TFTemperatureLogitsWarper,
            TFTopKLogitsWarper,
            TFTopPLogitsWarper,
        )
        from .tf_utils import (
            TFBeamSampleDecoderOnlyOutput,
            TFBeamSampleEncoderDecoderOutput,
            TFBeamSearchDecoderOnlyOutput,
            TFBeamSearchEncoderDecoderOutput,
            TFContrastiveSearchDecoderOnlyOutput,
            TFContrastiveSearchEncoderDecoderOutput,
            TFGenerationMixin,
            TFGreedySearchDecoderOnlyOutput,
            TFGreedySearchEncoderDecoderOutput,
            TFSampleDecoderOnlyOutput,
            TFSampleEncoderDecoderOutput,
        )

    try:
        if not is_flax_available():
            raise OptionalDependencyNotAvailable()
    except OptionalDependencyNotAvailable:
        pass
    else:
        from .flax_logits_process import (
            FlaxForcedBOSTokenLogitsProcessor,
            FlaxForcedEOSTokenLogitsProcessor,
            FlaxForceTokensLogitsProcessor,
            FlaxLogitsProcessor,
            FlaxLogitsProcessorList,
            FlaxLogitsWarper,
            FlaxMinLengthLogitsProcessor,
            FlaxNoRepeatNGramLogitsProcessor,
            FlaxSuppressTokensAtBeginLogitsProcessor,
            FlaxSuppressTokensLogitsProcessor,
            FlaxTemperatureLogitsWarper,
            FlaxTopKLogitsWarper,
            FlaxTopPLogitsWarper,
            FlaxWhisperTimeStampLogitsProcessor,
        )
        from .flax_utils import (
            FlaxBeamSearchOutput,
            FlaxGenerationMixin,
            FlaxGreedySearchOutput,
            FlaxSampleOutput,
        )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__, globals()["__file__"], _import_structure, module_spec=__spec__
    )

class GenerationMixin:
    """A class containing all functions for auto-regressive text genertation to be used as a mixin in [`PreTrainedModel`].
    The class exposes [`~geenration.GenerationMixin.generate`], which can be used for:
        - *greedy decoding* if `num_beams=1` and `do_sample=False`
        - *contrastive search* if `penalty_alpha>0` and `top_k>1`
        - *multinomial sampling* if `num_beams=1` and `do_sample=True`
        - *beam-search decoding* if `num_beams>1` and `do_sample=False`
        - *diverse beam-search decoding* if `num_beams>1` and `num_beam_groups>1`
        - *constrained beam-serach decoding* if `constraint!=None` or `forece_words_ids!=None`
        - *assisted decoding* if `assistant_model` or `prompt_lookup_num_tokens` is passed to `.generate()`
    
    To learn more about decoding strategies refer to the the [text generation strateigies guide]
    """
    def prepare_inputs_for_generation(
            self,
            input_ids: torch.LongTensor,
            past_key_values: Optional[Cache] = None,
            attention_mask: Optional[torch.LongTensor]=None,
            inputs_embeds: Optional[torch.FloatTensor]=None,
            cache_position:Optional[torch.LongTensor]=None,
            **kwargs
    ):
        """Prepare the model inputs for generation. In Includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.
        Args:
            input_ids (torch.LongTensor): _description_
            past_key_values (Optional[Cache], optional): _description_. Defaults to None.
            attention_mask (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            input_embeds (Optional[torch.FloatTensor], optional): _description_. Defaults to None.
            cache_position (Optional[torch.LongTensor], optional): _description_. Defaults to None.
        """
        model_inputs = {}
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None and input_ids.shape[1]==0:
                inputs_embeds = inputs_embeds[:, -cache_position.shape[0]:]
            elif (inputs_embeds is not None or (is_torchdynamo_compiling() or cache_position[-1] >= input_ids.shape[1])):
                input_ids =  input_ids[:, -cache_position.shape[0]:]
            elif input_ids.shape[1] == cache_position.shape[0]:
                input_ids = input_ids[:, cache_position]
        
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and len(cache_position) == inputs_embeds.shape[1]:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
        
        attention_mask = kwargs.pop("decoder_attention_mask", None) if self.config.is_encoder_decoder else attention_mask
        attention_mask_key = "decoder_attention_mask" if self.config.is_encoder_decoder else "attention_mask"
        position_ids_key = "decoder_position_ids" if self.config.is_encoder_decoder else "position_ids"
        if attention_mask is not None and kwargs.get(position_ids_key) is None and position_ids_key in set(inspect.signature(self.forward).parameters.keys()):
            position_ids = attention_mask.long().cumsum(-1) -1
            position_ids.masked_fill_(attention_mask==0,1)
            kwargs[position_ids_key] = position_ids
        for model_input_name in ["position_ids", "token_type_ids", "decoder_position_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values is not None:
                    current_input_length = model_inputs["inputs_embeds"].shape[1] if model_inputs.get("inputs_embeds") is not None else model_inputs[input_ids].shape[1]
                    model_input = model_input[:, -current_input_length:]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input
        
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim==2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device
            
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(self, "_prepare_4d_causal_attention_mask_with_cache_position", None)
            else:
                causal_mask_creation_function = getattr(self, "_prepare_4d_causal_attention_mask_with_cache_position", None)
            if causal_mask_creation_function is None:
                logger.warning_once(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` methond"
                    "defined in its base modelling class. Compiled forward passes will be sub-optimal"
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            
