import math
from collections import OrderedDict
import torch
from packaging import version
from torch import Tensor, nn
from .utils import logging

logger = logging.get_logger(__name__)

class PytorchGELUTanh(nn.Module):
    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            raise ImportError(f"You are using torch=={torch.__version__} but torch>=1.12.0 is required to use "
                              "PytorchGELUTanh. Please upgrade torch.")
        
    def forward(self, input:Tensor) -> Tensor:
        return nn.functional.gelu(input, approximate="tanh")
    
class NewGELUActivation(nn.Module):
    def forward(self, input:Tensor) -> Tensor:
        return 0.5 * input *(1.0 + torch.tanh(math.sqrt(2.0/math.pi) *(input + 0.044715 * torch.pow(input, 3.0))))
    
class GELUActivation(nn.Module):
    def __init__(self, use_gelu_python:bool=False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu
    
    def _gelu_python(self, input:Tensor) -> Tensor:
        return 0.5 * input *(1.0 + torch.erf(input/math.sqrt(2.0)))
    
    def forward(self, input:Tensor) -> Tensor:
        return self.act(input)
    
class FastGELUActivation(nn.Module):
    def forward(self, input:Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input*0.79788) * (1.0 +0.044715*input * input))
    

class QuickGELUActivation(nn.Module):
    def forward(self, input:Tensor) -> Tensor:
        return input * torch.sigmoid(1.702*input)
    

class ClippedGELUActivation(nn.Module):
    def __init__(self, min:float, max:float):
        if min > max:
            raise ValueError("min should be < max (got min: {min}, max: {max})")
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x:Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)

class AccurateGELUActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2/math.pi)

    def forward(self, input:Tensor) -> Tensor:
        return 0.5*input *(1+torch.tanh(self.precomputed_constant*(input+0.044715*torch.pow(input,3))))
    
class MishActivation(nn.Module):
    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish
    
    def _mish_python(self, input:Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))
    
    def forward(self, input:Tensor)->Tensor:
        return self.act(input)
    
class LaplaceActivation(nn.Module):
    def forward(self, input,mu=0.707107, sigma=0.282095):
        input = (input-mu).div(sigma * math.sqrt(2.0))
        return 0.5 *(1.0 + torch.erf(input))
class ReLUSquareActivation(nn.Module):
    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared
    
class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input
    
ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max":10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_relu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquareActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "prelu": nn.PReLU
}
ACT2FN = ClassInstantier(ACT2CLS)
def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")
    
gelu_python = get_activation("gelu_python")