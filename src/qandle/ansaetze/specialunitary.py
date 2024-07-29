from typing import List
import torch
import qandle.ansaetze.ansatz as ansatz
import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm

__all__ = ["SpecialUnitary"]


class SpecialUnitary(ansatz.UnbuiltAnsatz):
    def __init__(self, num_qubits: int, reps: int = 1, rotations=["ry"]):
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        self.rotations = [utils.parse_rot(r) for r in rotations]

    def build(self, *args, **kwargs) -> ansatz.BuiltAnsatz:
        return SpecialUnitaryBuilt(
            num_qubits=self.num_qubits,
            reps=self.reps,
            rotations=self.rotations,
        )

    def __str__(self) -> str:
        return "SU"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return [g.to_qasm() for g in self.decompose()] # type: ignore

    def decompose(self) -> List[op.Operator]:
        outp = []
        for r in self.rotations:
            for w in range(self.num_qubits):
                outp.append(r(w))
        for _ in range(self.reps):
            for w in range(self.num_qubits - 1):
                outp.append(op.CNOT(w, w + 1))
            for r in self.rotations:
                for w in range(self.num_qubits):
                    outp.append(r(w))
        return outp


class SpecialUnitaryBuilt(ansatz.BuiltAnsatz):
    def __init__(self, num_qubits: int, reps: int, rotations: list):
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        self.rotations = rotations
        layers = []
        for rep in range(reps + 1):
            block = []
            for w in range(num_qubits):
                for r in rotations:
                    block.append(r(w))
            layers.append(block)
        self.register_buffer(
            "cnots", self._get_cnot_matrix(num_qubits).contiguous(), persistent=False
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(*[la.build(num_qubits=num_qubits) for la in lay])
                for lay in layers
            ]
        )

    @staticmethod
    def _get_cnot_matrix(num_qubits: int):
        cnots = [
            op.BuiltCNOT._calculate_matrix(c=c, t=c + 1, num_qubits=num_qubits)
            for c in range(num_qubits - 1)
        ]
        r = cnots[0]
        for c in cnots[1:]:
            r = torch.matmul(r, c)
        return r

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = self.layers[0](state)
        for layer in self.layers[1:]:
            state = torch.matmul(self.cnots, state)
            state = layer(state)
        return state

    def __str__(self) -> str:
        return "SU"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return [g.to_qasm() for g in self.decompose()] # type: ignore

    def decompose(self) -> List[op.Operator]:
        res = []
        res.extend(self.layers[0])
        for mod in self.layers[1:]:
            res.extend(mod)
            for w in range(self.num_qubits - 1):
                res.append(op.CNOT(w, w + 1))
        return res
