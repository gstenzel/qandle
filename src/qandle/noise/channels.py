"""Noise channel definitions.

This module implements a small library of common quantum noise channels.  Only a
single fully functional class (`BitFlip`) is provided.  All other
channels contain ``TODO`` markers where their implementation should live.

The channels follow the same build pattern as gates in :mod:`qandle.operators`.
Unbuilt channels are lightweight containers that can be converted to torch
modules via :meth:`build`.
"""

from __future__ import annotations

from typing import Any

import torch

from .. import utils_gates, qasm
from ..operators import BuiltOperator, UnbuiltOperator

__all__ = [
    "BitFlip",
    "BitFlipChannel",  # backwards compatibility
    "PhaseFlip",
    "Depolarizing",
    "AmplitudeDamping",
    "PhaseDamping",
    "CorrelatedDepolarizing",
]

class BuiltBitFlip(BuiltOperator):
    def __init__(self, qubit: int, p: float, num_qubits: int):
        super().__init__()
        self.qubit = qubit
        self.p = float(p)
        self.num_qubits = num_qubits
        self._x = utils_gates.X(qubit, num_qubits)

    def __str__(self) -> str:
        return f"BitFlip(p={self.p})_{self.qubit}"

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        flipped = self._x(state)
        out = (1 - self.p) * state + self.p * flipped
        norm = torch.linalg.norm(out, dim=-1, keepdim=True)
        if norm.numel() == 1:
            norm = norm + 1e-12
        out = out / norm
        return out

    def to_matrix(self, **kwargs) -> torch.Tensor:
        i = torch.eye(2 ** self.num_qubits, dtype=torch.cfloat)
        x = self._x.to_matrix(**kwargs)
        return (1 - self.p) * i + self.p * x

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"// bit flip p={self.p}")

class BitFlip(UnbuiltOperator):
    def __init__(self, p: float, qubit: int):
        self.p = float(p)
        self.qubit = qubit

    def __str__(self) -> str:
        return f"BitFlip(p={self.p})_{self.qubit}"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return qasm.QasmRepresentation(gate_str=f"// bit flip p={self.p}")

    def build(self, num_qubits: int, **kwargs) -> BuiltBitFlip:
        return BuiltBitFlip(qubit=self.qubit, p=self.p, num_qubits=num_qubits)

# Backwards compatibility
BitFlipChannel = BitFlip


class PhaseFlip(UnbuiltOperator):
    """Z-basis dephasing channel.

    TODO: implement phase flip noise.
    """

    def __init__(self, p: float, qubit: int):
        self.p = float(p)
        self.qubit = qubit

    def build(self, num_qubits: int, **kwargs) -> BuiltOperator:
        # TODO: return built implementation
        raise NotImplementedError


class Depolarizing(UnbuiltOperator):
    """Single-qubit depolarizing channel."""

    def __init__(self, p: float, qubit: int):
        self.p = float(p)
        self.qubit = qubit

    def build(self, num_qubits: int, **kwargs) -> BuiltOperator:
        # TODO: implement
        raise NotImplementedError


class AmplitudeDamping(UnbuiltOperator):
    """Amplitude damping noise channel."""

    def __init__(self, gamma: float, qubit: int):
        self.gamma = float(gamma)
        self.qubit = qubit

    def build(self, num_qubits: int, **kwargs) -> BuiltOperator:
        # TODO: implement
        raise NotImplementedError


class PhaseDamping(UnbuiltOperator):
    """Phase damping channel."""

    def __init__(self, gamma: float, qubit: int):
        self.gamma = float(gamma)
        self.qubit = qubit

    def build(self, num_qubits: int, **kwargs) -> BuiltOperator:
        # TODO: implement
        raise NotImplementedError


class CorrelatedDepolarizing(UnbuiltOperator):
    """Two-qubit correlated depolarizing channel."""

    def __init__(self, p: float, qubits: tuple[int, int]):
        self.p = float(p)
        self.qubits = qubits

    def build(self, num_qubits: int, **kwargs) -> BuiltOperator:
        # TODO: implement
        raise NotImplementedError

