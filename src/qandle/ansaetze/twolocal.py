import torch
import typing
import copy

import qandle.config as config
import qandle.operators as op
import qandle.qasm as qasm
import qandle.ansaetze.ansatz as ansatz

__all__ = ["TwoLocal"]


class TwoLocal(ansatz.UnbuiltAnsatz):
    """
    The 2-local circuit as described in `the Qiskit docs <https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.TwoLocal>`_. It comprises alternating layers of single qubit rotations and CNOT gates. This implementation uses the :code:`linear` entanglement map.
    """

    def __init__(
        self,
        qubits: typing.Union[typing.List[int], None] = None,
        depth: int = 1,
        remapping: typing.Union[typing.Callable, None] = config.DEFAULT_MAPPING,
        q_params=None,
    ):
        super().__init__()
        self.depth = depth
        self.remapping = remapping
        self.q_params = q_params
        self.qubits = qubits

    def build(self, *args, **kwargs) -> "ansatz.BuiltAnsatz":
        return TwoLocalBuilt(
            num_qubits=kwargs["num_qubits"],
            qubits=self.qubits,
            depth=self.depth,
            remapping=self.remapping,
            q_params=self.q_params,
        )

    def __str__(self) -> str:
        return "TL"

    def to_qasm(self) -> str:
        return [g.to_qasm() for g in self.decompose()]  # type: ignore

    def decompose(self) -> typing.List[op.Operator]:
        if self.qubits is None:
            raise ValueError(
                "It is not specified on which qubits to apply the ansatz. Build the circuit to specify or pass them explicitly as a list"
            )
        res = []
        qp = (
            self.q_params if self.q_params is not None else torch.rand(self.depth, len(self.qubits))
        )
        for d in range(self.depth):
            for wi in range(len(self.qubits)):
                res.append(op.RY(qubit=self.qubits[wi], theta=qp[d, wi]))
            for wi in range(len(self.qubits) - 1):
                res.append(op.CNOT(self.qubits[wi], self.qubits[wi + 1]))
        return res


class TwoLocalBuilt(ansatz.BuiltAnsatz):
    def __init__(
        self,
        num_qubits: int,
        qubits: typing.Union[typing.List[int], None],
        depth: int = 1,
        remapping: typing.Union[typing.Callable, None] = config.DEFAULT_MAPPING,
        q_params=None,
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.remapping = remapping
        self.qubits = qubits or list(range(num_qubits))
        if q_params is None or q_params.shape != (depth, len(self.qubits)):
            q_params = torch.rand(depth, num_qubits)
        self.register_buffer(
            "cnot_matrices", self._get_cnot_matrix(self.qubits, self.num_qubits), persistent=False
        )
        layers = [
            [op.RY(qubit=w, theta=q_params[d, w]).build(num_qubits) for w in range(num_qubits)]
            for d in range(depth)
        ]
        self.mods = torch.nn.ModuleList([torch.nn.Sequential(*lay) for lay in layers])

    @staticmethod
    def _get_cnot_matrix(qubits: typing.List[int], num_qubits: int):
        cnots = [
            op.BuiltCNOT._calculate_matrix(c=qubits[ci], t=qubits[ci + 1], num_qubits=num_qubits)
            for ci in range(len(qubits) - 1)
        ]
        r = cnots[0]
        for c in cnots[1:]:
            r = torch.matmul(r, c)
        return r

    def forward(self, state: torch.Tensor):
        for d in range(self.depth):
            state = self.mods[d](state)
            state = self.cnot_matrices @ state
        return state

    def __str__(self) -> str:
        return "TL"

    def to_qasm(self) -> qasm.QasmRepresentation:
        outp = []
        for d in range(self.depth):
            outp.extend([g.to_qasm() for g in self.mods[d]])
            for ci in range(len(self.qubits) - 1):
                outp.append(op.CNOT(self.qubits[ci], self.qubits[ci + 1]).to_qasm())
        return outp  # type: ignore

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        outp = []
        for mod in self.mods:
            for seq in mod:
                outp.append(copy.deepcopy(seq))
            for ci in range(len(self.qubits) - 1):
                outp.append(op.CNOT(self.qubits[ci], self.qubits[ci + 1]))
        return outp
