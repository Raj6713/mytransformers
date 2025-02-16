from typing import TYPE_CHECKING
from ...utils import _LazyModule
from ...utils.import_utils import define_impore_structure

if TYPE_CHECKING:
    from .configuration_gpt2 import *
    from .modeling_flax_gpt2 import *
    from .modeling_gpt2 import *
    from .modeling_tf_gpt2 import *
    from .tokenization_gpt2 import *
    from .tokenization_gpt2_fast import *
    from .tokenization_gpt2_tf import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(
        __name__, _file, define_impore_structure(_file), moduele_space=__spec__
    )
