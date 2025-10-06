import math
import torch
from .operators import U

__all__ = ["H", "X", "Y", "Z", "S", "T"]

SQRT2 = math.sqrt(2)


def H(qubit: int, num_qubits: int | None = None):
    """Hadamard gate on a single qubit."""
    num_qubits = qubit + 1 if num_qubits is None else num_qubits
    mat = torch.tensor([[1, 1], [1, -1]], dtype=torch.complex64) / SQRT2
    return U(qubit=qubit, matrix=mat).build(num_qubits=num_qubits)


def X(qubit: int, num_qubits: int | None = None):
    """Pauli-X gate (bit flip) on a single qubit."""
    num_qubits = qubit + 1 if num_qubits is None else num_qubits
    mat = torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)
    return U(qubit=qubit, matrix=mat).build(num_qubits=num_qubits)


def Y(qubit: int, num_qubits: int | None = None):
    """Pauli-Y gate (bit and phase flip) on a single qubit."""
    num_qubits = qubit + 1 if num_qubits is None else num_qubits
    mat = torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64)
    return U(qubit=qubit, matrix=mat).build(num_qubits=num_qubits)


def Z(qubit: int, num_qubits: int | None = None):
    """Pauli-Z gate (phase flip) on a single qubit."""
    num_qubits = qubit + 1 if num_qubits is None else num_qubits
    mat = torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    return U(qubit=qubit, matrix=mat).build(num_qubits=num_qubits)


def S(qubit: int, num_qubits: int | None = None):
    """Phase gate (S gate) on a single qubit."""
    num_qubits = qubit + 1 if num_qubits is None else num_qubits
    mat = torch.tensor([[1, 0], [0, 1j]], dtype=torch.complex64)
    return U(qubit=qubit, matrix=mat).build(num_qubits=num_qubits)


def T(qubit: int, num_qubits: int | None = None):
    """T gate (π/8 gate) on a single qubit."""
    num_qubits = qubit + 1 if num_qubits is None else num_qubits
    val = complex(math.cos(math.pi / 4), math.sin(math.pi / 4))
    mat = torch.tensor([[1, 0], [0, val]], dtype=torch.complex64)
    return U(qubit=qubit, matrix=mat).build(num_qubits=num_qubits)
