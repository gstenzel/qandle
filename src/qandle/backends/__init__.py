import torch
from typing import Protocol, runtime_checkable, Sequence


@runtime_checkable
class QuantumBackend(Protocol):
    '''Backend interface.'''

    def allocate(self, n_qubits: int): ...

    def apply_1q(self, gate: torch.Tensor, q: int): ...

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int): ...

    def measure(self, qubits: Sequence[int] | None = None) -> torch.Tensor: ...


__all__ = [
    "QuantumBackend",
    "StateVectorBackend",
    "MPSBackend",
]

from .statevector import StateVectorBackend
from .mps import MPSBackend
