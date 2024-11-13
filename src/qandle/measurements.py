import torch

from . import operators as op
from . import qasm

from .errors import UnbuiltGateError
import random

import warnings


class BuiltMeasurement(op.BuiltOperator):
    def __str__(self) -> str:
        return "M"

    def to_qasm(self) -> qasm.QasmRepresentation:
        f = "measure q[{}] -> c[{}]"
        i = hash(random.random())
        return [
            qasm.QasmRepresentation(gate_str=f.format(w, w), qasm3_outputs=f"measured_{i}_{w}")
            for w in range(self.num_qubits)
        ]  # type: ignore

    def decompose(self):
        warnings.warn(
            "Decomposed measurements a no-op. Consider acting on the resulting statevector directly",
            RuntimeWarning,
        )

        class SingleQubitMeasurementDummy(op.UnbuiltOperator, torch.nn.Module):
            def __init__(self, qubit: int):
                self.qubit = qubit
                super().__init__()

            def __str__(self) -> str:
                return "M"

            def to_qasm(self) -> qasm.QasmRepresentation:
                return qasm.QasmRepresentation(
                    gate_str=f"measure q[{self.qubit}] -> c[{self.qubit}]",
                    qasm3_outputs=f"measured_{hash(random.random())}_{self.qubit}",
                )

            def forward(self, state: torch.Tensor) -> torch.Tensor:
                warnings.warn(
                    "Forward on a decomposed measurement is a no-op. Consider acting on the resulting statevector directly"
                )
                return state

            def build(self, num_qubits: int, **kwargs):
                return self

        return [SingleQubitMeasurementDummy(w) for w in range(self.num_qubits)]

    def to_matrix(self):
        """
        No-op for measurements.
        """
        return torch.eye(2**self.num_qubits)


class UnbuiltMeasurement(op.UnbuiltOperator):
    def __str__(self) -> str:
        return "M"

    def to_qasm(self) -> str:
        raise UnbuiltGateError(f"Unbuilt {self.__class__.__name__} cannot be converted to qasm.")

    def build(self, num_qubits: int, **kwargs) -> BuiltMeasurement:
        return BUILT_CLASS_RELATION[self.__class__](num_qubits=num_qubits, **kwargs)  # type: ignore


class MeasureAllAbsolute(BuiltMeasurement):
    """
    Measure the absolute probability of the state-vector.
    """

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state.abs().pow(2)


MeasureJointProbability = MeasureAllAbsolute  # alias


class MeasureState(BuiltMeasurement):
    """
    Measure the state-vector. As the state-vector is already a known, this measurement is a no-op.
    """

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return state


class MeasureProbability(UnbuiltMeasurement):
    r"""
    Measure the probability of each qubit being in the :math:`\ket{0}` state.
    Use :code:`1 - measure_probability` to get the probability of each qubit being in the :math:`\ket{1}` state.
    """


class MeasureProbabilityBuilt(BuiltMeasurement):
    def __init__(self, num_qubits: int):
        super().__init__()
        BuiltMeasurement.__init__(self)
        self.num_qubits = num_qubits
        self.qubits = list(range(num_qubits))
        indices0 = self._get_indices0()
        indices1 = ~indices0
        self.register_buffer("indices0", indices0.contiguous(), persistent=False)
        # currently unused
        self.register_buffer("indices1", indices1.contiguous(), persistent=False)

    def _get_indices0(self):
        indices = []
        for w in range(self.num_qubits):
            ind = (torch.arange(2**self.num_qubits) // (2**w)) % 2 == 0
            indices.append(ind)
        return torch.stack(indices, dim=0).flip(dims=(0,))

    def forward(self, state: torch.Tensor):
        rep = state.repeat(self.num_qubits, 1, 1)
        sums = rep * self.indices0[:, None, :]
        return (sums.abs().pow(2).sum(dim=-1).squeeze()).transpose(-1, 0)


BUILT_CLASS_RELATION = {MeasureProbability: MeasureProbabilityBuilt}
