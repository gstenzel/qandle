from __future__ import annotations
import torch
import math
from typing import List, Tuple, Dict, Any

"""
Parameter-shift gradients for circuits built from RX, RY and RZ gates.
"""

_SHIFT = math.pi / 2
_COFF = 0.5


def _eval_circuit(
    circuit: torch.nn.Module,
    state: torch.Tensor | None,
    kwargs: Dict[str, Any],
) -> torch.Tensor:
    """Evaluate *circuit* with autograd disabled and return a detached scalar."""
    with torch.no_grad():
        out = circuit(state, **kwargs) if state is not None else circuit(**kwargs)
    if out.numel() != 1:
        raise RuntimeError(
            "Parameter-shift path expects the circuit to return a scalar "
            f"(e.g. an expectation value). Got shape {tuple(out.shape)}."
        )
    return out.squeeze().to(dtype=torch.float32)


class _ParameterShiftFn(torch.autograd.Function):
    """Autograd function implementing the parameter-shift rule."""

    @staticmethod
    def forward(ctx, *args):
        *params, circuit, state, kwargs = args
        ctx.circuit = circuit
        ctx.state = state
        ctx.kwargs = kwargs
        ctx.save_for_backward(*params)
        return _eval_circuit(circuit, state, kwargs)

    @staticmethod
    def backward(ctx, *grad_output):
        grad_out = grad_output[0]
        params: Tuple[torch.Tensor, ...] = ctx.saved_tensors
        circuit, state, kwargs = ctx.circuit, ctx.state, ctx.kwargs

        grads: List[torch.Tensor] = []
        with torch.inference_mode():
            for p in params:
                grad_for_p = torch.zeros_like(p)
                flat = p.view(-1)
                for idx in range(flat.numel()):
                    original = flat[idx].item()

                    flat[idx] = original + _SHIFT
                    f_plus = _eval_circuit(circuit, state, kwargs)

                    flat[idx] = original - _SHIFT
                    f_minus = _eval_circuit(circuit, state, kwargs)

                    flat[idx] = original
                    grad_val = _COFF * (f_plus - f_minus)
                    grad_for_p.view(-1)[idx] = grad_val

                grads.append(grad_out * grad_for_p)

        grads.extend([None, None, None])
        return tuple(grads)


def parameter_shift_forward(
    circuit: torch.nn.Module,
    state: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    """Execute *circuit* while enabling parameter-shift gradients."""
    params = [p for p in circuit.parameters() if p.requires_grad]
    return _ParameterShiftFn.apply(*params, circuit, state, kwargs)

