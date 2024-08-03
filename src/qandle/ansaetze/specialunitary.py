import typing
import torch
import qandle.ansaetze.ansatz as ansatz
import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm

__all__ = ["SpecialUnitary"]


class SpecialUnitary(ansatz.UnbuiltAnsatz):
    """
    The hardware efficient SU(2) 2-local circuit as described in `the Qiskit docs <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.EfficientSU2>`_.
    """

    def __init__(
        self,
        qubits: typing.Union[typing.List[int], None] = None,
        reps: int = 1,
        rotations=["ry"],
    ):
        super().__init__()
        self.qubits = qubits
        self.reps = reps
        self.rotations = [utils.parse_rot(r) for r in rotations]

    def build(self, *args, **kwargs) -> ansatz.BuiltAnsatz:
        return SpecialUnitaryBuilt(
            num_qubits=kwargs["num_qubits"],
            reps=self.reps,
            rotations=self.rotations,
            qubits=self.qubits,
        )

    def __str__(self) -> str:
        return "SU"

    def to_qasm(self) -> qasm.QasmRepresentation:
        return [g.to_qasm() for g in self.decompose()]  # type: ignore

    def decompose(self) -> typing.List[op.Operator]:
        if self.qubits is None:
            raise ValueError(
                "It is not specified on which qubits to apply the ansatz. Build the circuit to specify or pass them explicitly as a list"
            )
        outp = []
        for r in self.rotations:
            for w in self.qubits:
                outp.append(r(w))
        for _ in range(self.reps):
            for wi in range(len(self.qubits) - 1):
                outp.append(op.CNOT(self.qubits[wi], self.qubits[wi + 1]))
            for r in self.rotations:
                for w in self.qubits:
                    outp.append(r(w))
        return outp


class SpecialUnitaryBuilt(ansatz.BuiltAnsatz):
    def __init__(
        self,
        num_qubits: int,
        qubits: typing.Union[typing.List[int], None],
        reps: int,
        rotations: typing.List[op.Operator],
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.reps = reps
        self.rotations = rotations
        self.qubits = qubits or list(range(num_qubits))
        self.to_matrix, self.to_state = utils.get_matrix_transforms(num_qubits, self.qubits)
        layers = []
        for rep in range(reps + 1):
            block = []
            for w in self.qubits:
                for r in rotations:
                    block.append(r(w))
            layers.append(block)
        self.register_buffer(
            "cnots",
            self._get_cnot_matrix(qubits=self.qubits, num_qubits=num_qubits).contiguous(),
            persistent=False,
        )
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(*[la.build(num_qubits=num_qubits) for la in lay])
                for lay in layers
            ]
        )

    @staticmethod
    def _get_cnot_matrix(num_qubits: int, qubits: typing.List[int]):
        cnots = [
            op.BuiltCNOT._calculate_matrix(c=qubits[c], t=qubits[c + 1], num_qubits=num_qubits)
            for c in range(len(qubits) - 1)
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
        return [g.to_qasm() for g in self.decompose()]  # type: ignore

    def decompose(self) -> typing.List[op.Operator]:
        res = []
        res.extend(self.layers[0])
        for mod in self.layers[1:]:
            res.extend(mod)
            for w in range(len(self.qubits) - 1):
                res.append(op.CNOT(self.qubits[w], self.qubits[w + 1]))
        return res
