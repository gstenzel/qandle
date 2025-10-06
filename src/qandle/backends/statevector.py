import torch
import math
from typing import Sequence
from . import QuantumBackend

class StateVectorBackend(QuantumBackend):
    """State-vector simulator."""

    def __init__(self, n_qubits: int, dtype=torch.complex64, device="cpu"):
        self.dtype = dtype
        self.device = device
        self.allocate(n_qubits)

    def allocate(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.state = torch.zeros(2 ** n_qubits, dtype=self.dtype, device=self.device)
        self.state[0] = 1
        return self

    def _apply_gate_dense(self, gate: torch.Tensor, qubits: Sequence[int]):
        gate = gate.to(self.state)
        n = self.n_qubits
        q_be = [n - q - 1 for q in qubits]
        perm = q_be + [i for i in range(n) if i not in q_be]
        inv_perm = [perm.index(i) for i in range(n)]
        psi = self.state.view([2] * n).permute(perm).reshape(2 ** len(qubits), -1)
        psi = torch.matmul(gate, psi)
        psi = psi.reshape([2] * len(qubits) + [2] * (n - len(qubits)))
        psi = psi.permute(inv_perm).reshape(2 ** n)
        self.state = psi

    def apply_1q(self, gate: torch.Tensor, q: int):
        if gate.shape[0] != 2:
            n = int(math.log2(gate.shape[0]))
            idx0 = 0
            idx1 = 1 << (n - q - 1)
            gate = gate[[idx0, idx1]][:, [idx0, idx1]]
        self._apply_gate_dense(gate, [q])

    def apply_2q(self, gate: torch.Tensor, q1: int, q2: int):
        if gate.shape[0] != 4:
            n = int(math.log2(gate.shape[0]))
            idx = lambda b1, b2: ((b1 << (n - q1 - 1)) | (b2 << (n - q2 - 1)))
            sel = [idx(0,0), idx(0,1), idx(1,0), idx(1,1)]
            gate = gate[sel][:, sel]
        self._apply_gate_dense(gate, [q1, q2])

    def apply_dense(self, gate: torch.Tensor, qubits: Sequence[int]):
        self._apply_gate_dense(gate, qubits)

    def measure(self, qubits: Sequence[int] | None = None) -> torch.Tensor:
        probs = self.state.abs() ** 2
        if qubits is None:
            return probs
        n = self.n_qubits
        out = torch.zeros(2 ** len(qubits), dtype=probs.dtype, device=probs.device)
        for i, p in enumerate(probs):
            key = 0
            for j, q in enumerate(qubits):
                bit = (i >> (n - q - 1)) & 1
                key |= bit << j
            out[key] += p
        return out
