from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True, slots=True)
class GradientState:
    norm: float
    clipped: bool
    parameter_tensors: int

    def describe(self):
        return asdict(self)


def inspect_gradients(model, max_norm: float = 0.0) -> GradientState:
    parameters = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if not parameters:
        raise RuntimeError('The objective produced no parameter gradients')
    if max_norm:
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    else:
        norms = [parameter.grad.detach().norm(2) for parameter in parameters]
        norm = torch.stack(norms).norm(2)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite parameter gradient norm')
    value = float(norm)
    return GradientState(value, bool(max_norm and value > max_norm), len(parameters))
