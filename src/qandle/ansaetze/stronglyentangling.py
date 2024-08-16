import torch
import typing
import qw_map
import copy


import qandle.config as config
import qandle.operators as op
import qandle.utils as utils
import qandle.qasm as qasm
from qandle.ansaetze.ansatz import UnbuiltAnsatz, BuiltAnsatz


__all__ = ["StronglyEntanglingLayer"]


class StronglyEntanglingLayer(UnbuiltAnsatz):
    """
    A strongly entangling layer, inspired by `this paper <https://arxiv.org/abs/1804.00633>`_.
    Consists of a series of single-qubit rotations on each qubit specified in the :code:`qubits` parameter, followed by a series of CNOT gates. A higher :code:`depth` will increase the expressivity of the circuit at the cost of more parameters and a linear increase in runtime.
    """

    def __init__(
        self,
        qubits: typing.List[int],
        num_qubits_total: typing.Union[int, None] = None,
        depth: int = 1,
        rotations=["rz", "ry", "rz"],
        q_params=None,
        remapping: typing.Union[typing.Callable, None] = config.DEFAULT_MAPPING,
    ):
        super().__init__()
        self.depth = depth
        self.qubits = qubits
        self.num_qubits = num_qubits_total
        self.rots = [utils.parse_rot(r) for r in rotations]
        if q_params is None:
            q_params = torch.rand(depth, len(qubits), len(rotations))
        self.q_params = q_params
        if remapping is None:
            remapping = qw_map.none
        self.remapping = remapping

    def build(self, *args, **kwargs) -> BuiltAnsatz:
        return StronglyEntanglingLayerBuilt(
            num_qubits=kwargs["num_qubits"],
            depth=self.depth,
            rotations=self.rots,
            q_params=self.q_params,
            remapping=self.remapping,
            qubits=self.qubits,
        )

    def __str__(self) -> str:
        return "SEL"

    def to_qasm(self) -> qasm.QasmRepresentation:
        raise NotImplementedError("TODO")

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        layers = []
        for d in range(self.depth):
            for wi, w in enumerate(self.qubits):
                for r in range(len(self.rots)):
                    layers.append(
                        self.rots[r](
                            qubit=w,
                            theta=self.q_params[d, wi, r],
                            remapping=self.remapping,
                        )
                    )
            layers.extend(
                StronglyEntanglingLayerBuilt._get_cnots(
                    self.qubits, self.num_qubits, d % (self.num_qubits - 1)
                )
            )
        return layers


class StronglyEntanglingLayerBuilt(BuiltAnsatz):
    def __init__(
        self,
        num_qubits: int,
        qubits: typing.List[int],
        depth: int,
        rotations: typing.List[op.UnbuiltParametrizedOperator],
        q_params: torch.Tensor,
        remapping: typing.Union[typing.Callable, None],
    ):
        super().__init__()
        self.num_qubits = num_qubits
        self.depth = depth
        self.rots = rotations
        self.qubits = qubits
        layers = []
        for d in range(depth):
            for wi, w in enumerate(self.qubits):
                for r in range(len(self.rots)):
                    layers.append(
                        self.rots[r](
                            qubit=w,
                            theta=q_params[d, wi, r],
                            remapping=remapping,  # type: ignore
                        ).build(num_qubits)
                    )
            layers.extend(self._get_cnots(self.qubits, self.num_qubits, d % (len(self.qubits) - 1)))
        self.mods = torch.nn.Sequential(*layers)

    def forward(self, state: torch.Tensor):
        return self.mods(state)

    @staticmethod
    def _get_cnots(
        qubits: typing.List[int], num_qubits_total: int, iteration: int
    ) -> typing.List[op.UnbuiltOperator]:
        assert iteration + 1 < num_qubits_total
        cnots = []
        for ci in range(len(qubits)):
            ti = (ci + iteration + 1) % len(qubits)
            cnots.append(op.CNOT(qubits[ci], qubits[ti]).build(num_qubits_total))
        return cnots

    def __str__(self) -> str:
        return "SEL"

    def to_qasm(self) -> str:
        return [g.to_qasm() for g in self.decompose()]  # type: ignore

    def decompose(self) -> typing.List[op.UnbuiltOperator]:
        layers = []
        for mod in self.mods:
            layers.append(copy.deepcopy(mod))
        return layers
